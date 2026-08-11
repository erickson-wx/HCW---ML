import sys
import os
import glob
import pickle
import requests
import struct
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf

from keras_unet_collection import models
from tensorflow.train import Example, Feature, Features, BytesList, Int64List
from custom_losses import WeightedMSE,WeightedLogMSE

def serialize_example(input_array, label_array):
    feature = {
            'input': Feature(bytes_list=BytesList(value=[input_array.tobytes()])),
            'label': Feature(bytes_list=BytesList(value=[label_array.tobytes()])),
            'input_shape': Feature(int64_list=Int64List(value=input_array.shape)),
            'label_shape': Feature(int64_list=Int64List(value=label_array.shape))
            }

    return Example(features=Features(feature=feature)).SerializeToString()

def write_to_tfrecord(input_stack, label_stack, output_path):
    with tf.io.TFRecordWriter(output_path) as writer:
        print('Shape of input stack is:', input_stack.shape)
        for t in range(input_stack.shape[0]):
            inputs = input_stack[t,:,:,:]
            labels = label_stack[t,:,:]
            example = serialize_example(input_array=inputs.astype(np.float32),label_array=labels.astype(np.float32))

            writer.write(example)

def preprocess_and_write_tfrecord(dir, tfrecord_out_dir, epoch='hist'):
    os.makedirs(tfrecord_out_dir, exist_ok=True)

    input_features = ['CAPE', 'CIN', 'SRH', 'APCP', 'MSLP', 'PW', 
                      'WSPD', 'WDIR', 'Q2', 'T2', 'U10', 'V10']

    feature_to_file = {
        'CAPE': 'CAPE', 'CIN': 'CIN', 'SRH': 'SRH', 'APCP': 'APCP',
        'MSLP': 'MSLP', 'PW': 'PW',
        'WSPD': 'WSPD', 'WDIR': 'WSPD',  # both from same file
        'Q2': 'Q2', 'T2': 'T2', 'U10': 'U10', 'V10': 'V10'
    }

    # Step 1: build file lists by file source (not by individual feature)
    file_lists = {}
    for file_var in set(feature_to_file.values()):
        file_lists[file_var] = sorted(glob.glob(os.path.join(dir, f'{file_var}*.nc')),
                                      key=lambda x: x[-10:-3])

    label_files = sorted(glob.glob(os.path.join(dir, 'W_UP_MAX*.nc')), key=lambda x: x[-10:-3])
    if epoch == 'fut':
        label_files = label_files[:16] # Only take first four years

    for i in range(len(label_files)):
        print(f'Running for {label_files[i][-10:-3]}')
        #print(f'Writing historical data set to TF at path data_{i:03d}.tfrecord...')
        #continue
        feature_arrays = []

        for feature in input_features:
            file_var = feature_to_file[feature]
            ds = xr.open_dataset(file_lists[file_var][i])
            if 'west_east' in ds.dims:
                ds = ds.rename({'west_east':'y','south_north':'x'})

            # Indices for indexing to get proper tile shape
            x1 = 54
            x2 = 566
            y1 = 54
            y2 = 566

            if file_var == 'WSPD' and feature in ['WSPD', 'WDIR']:
                # Assume the variable is something like 'WSPD_WDIR' with dimension 'wspd_wdir'
                wspd_var_name = list(ds.data_vars)[0]  # Assuming one variable, otherwise specify manually
                wspd_data = ds[wspd_var_name]

                if 'wspd_wdir' in wspd_data.dims:
                    if feature == 'WSPD':
                        arr = wspd_data.isel(wspd_wdir=0).transpose('x', 'y', 'time').values[x1:x2,y1:y2] # 
                    elif feature == 'WDIR':
                        arr = wspd_data.isel(wspd_wdir=1).transpose('x', 'y', 'time').values[x1:x2,y1:y2]
                else:
                    raise ValueError(f"Expected dimension 'wspd_wdir' in variable {wspd_var_name}")
            elif feature == 'MSLP':
                arr = ds['slp'].transpose('x','y','time').values[x1:x2,y1:y2]
            elif feature == 'PW':
                arr = ds['pw'].transpose('x','y','time').values[x1:x2,y1:y2]
            else:
                arr = ds[feature].transpose('x', 'y', 'time').values[x1:x2,y1:y2]
            print(f'Feature {feature} has shape ', arr.shape)
            feature_arrays.append(arr)

        # Stack all features as separate channels
        input_stack = np.stack(feature_arrays, axis=-1)  # shape: [x, y, time, feature]

        # Read label
        with xr.open_dataset(label_files[i]) as ds_label:
            label_stack = ds_label['W'].transpose('x', 'y', 'time').values[x1:x2,y1:y2]  # shape: [x, y, time]
        
        # Fill NaNs and normalize
        input_stack = np.nan_to_num(input_stack,nan=0)
        label_stack = np.nan_to_num(label_stack,nan=0)

        input_min = np.min(input_stack)
        input_max = np.max(input_stack)

        inputs = (input_stack - input_min) / (input_max - input_min) 

        if epoch == 'hist':
            # Threshold at 10 m/s
            print('Splitting into extreme and normal subsets...')
            threshold = 10.0

            extreme_idx = np.where(np.any(label_stack >= threshold, axis=(0,1)))[0]
            normal_idx = np.where(~np.any(label_stack >= threshold, axis=(0,1)))[0]

            X_ext, y_ext = inputs[:,:,extreme_idx,:],label_stack[:,:,extreme_idx]
            X_norm, y_norm = inputs[:,:,normal_idx,:],label_stack[:,:,normal_idx]

            X_ext = np.transpose(X_ext, (2,0,1,3))
            y_ext = np.transpose(y_ext, (2,0,1))

            X_norm = np.transpose(X_norm, (2,0,1,3))
            y_norm = np.transpose(y_norm, (2,0,1))

            # Write one normal and one extreme TFRecord per month
            print(f'Writing historical data set to TF at path data_{i:03d}.tfrecord...')
            normal_path = os.path.join(tfrecord_out_dir, f'norm_{i:03d}.tfrecord')
            extreme_path = os.path.join(tfrecord_out_dir, f'ext_{i:03d}.tfrecord')
            
            write_to_tfrecord(X_norm, y_norm, extreme_path)
            write_to_tfrecord(X_ext, y_ext, normal_path)
        else:
            print(f'Writing future data set to TF at path data_{i:03d}.tfrecord...')
            out_path = os.path.join(tfrecord_out_dir, f'data_{i:03d}.tfrecord')

            inputs = np.transpose(inputs, (2,0,1,3))
            label_stack = np.transpose(label_stack, (2,0,1))

            write_to_tfrecord(inputs, label_stack, out_path)

def parse_example(example_proto):
    features = {
        'input': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'input_shape': tf.io.VarLenFeature(tf.int64),
        'label_shape': tf.io.VarLenFeature(tf.int64)
    }
    parsed = tf.io.parse_single_example(example_proto, features)
    
    #input_shape = tf.sparse.to_dense(parsed['input_shape'])
    #label_shape = tf.sparse.to_dense(parsed['label_shape'])
    input_shape = (512,512,12)
    label_shape = (512,512)

    input_array = tf.io.decode_raw(parsed['input'], tf.float32)
    label_array = tf.io.decode_raw(parsed['label'], tf.float32)

    input_array = tf.reshape(input_array, input_shape)
    label_array = tf.reshape(label_array, label_shape)

    return input_array, label_array

def load_dataset(tfrecord_dir, ds, batch_size=8, shuffle=True):
    #files_read = sorted(glob.glob(os.path.join(tfrecord_dir,'{ds}*.tfrecord')),key=lambda x: x[-12:-10])
    #print('Files being loaded for dataset:',files_read)
    files = tf.data.Dataset.list_files(os.path.join(tfrecord_dir,f'{ds}*.tfrecord'),shuffle=False)
    #files = tf.data.Dataset.list_files(os.path.join(tfrecord_dir, f'{ds}*.tfrecord'),shuffle=False)
    dataset = files.interleave(tf.data.TFRecordDataset,
                               cycle_length=4,
                               num_parallel_calls=tf.data.AUTOTUNE,
                               deterministic=True)
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE,deterministic=True)
    print('Loaded TF dataset:',dataset)
    if shuffle:
        dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
import tensorflow as tf

def make_balanced_dataset(
    normal_ds,
    extreme_ds,
    batch_size,
    extremes_per_batch_ratio=0.5,
    shuffle_size=2048,
    num_batches_per_epoch=1000,
):
    """
    Creates a balanced dataset without using repeat(), to avoid memory buildup.

    Args:
        normal_ds: tf.data.Dataset of normal samples.
        extreme_ds: tf.data.Dataset of extreme samples.
        batch_size: total batch size.
        extremes_per_batch_ratio: fraction of each batch from extreme_ds.
        shuffle_size: buffer size for shuffling.
        num_batches_per_epoch: number of mixed batches per epoch.

    Returns:
        balanced_ds: shuffled, batched, prefetched dataset for one epoch.
    """

    # Shuffle each source dataset independently
    normal_ds = normal_ds.shuffle(shuffle_size, reshuffle_each_iteration=True)
    extreme_ds = extreme_ds.shuffle(shuffle_size, reshuffle_each_iteration=True)

    # Number of examples to draw per source this epoch
    n_extreme = int(num_batches_per_epoch * batch_size * extremes_per_batch_ratio)
    n_normal  = int(num_batches_per_epoch * batch_size * (1 - extremes_per_batch_ratio))

    # Sample a finite number of examples from each
    normal_sample = normal_ds.take(n_normal)
    extreme_sample = extreme_ds.take(n_extreme)

    # Combine them into a single dataset (balanced mix)
    mixed = tf.data.Dataset.sample_from_datasets(
        [extreme_sample, normal_sample],
        weights=[extremes_per_batch_ratio, 1 - extremes_per_batch_ratio],
        stop_on_empty_dataset=True
    )

    # Final batching and prefetching (no repeat!)
    balanced_ds = mixed.batch(batch_size, drop_remainder=True)
    balanced_ds = balanced_ds.prefetch(tf.data.AUTOTUNE)
    #balanced_ds = mixed #Note: If this doesn't work, uncomment above, readd lines to unbatch train/val

    return balanced_ds

def get_train_val_balanced_datasets(normal_ds,extreme_ds,val_fraction=0.1,batch_size=32,extremes_per_batch=6,shuffle_buffer=1024,seed=42,):
    # Shuffle to randomize frames
    normal_ds = normal_ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=False)
    extreme_ds = extreme_ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=False)

    def count_elements(ds):
        return sum(1 for _ in ds)

    num_normal = count_elements(normal_ds)
    num_extreme = count_elements(extreme_ds)

    n_val_normal = int(val_fraction * num_normal)
    n_val_extreme = int(val_fraction * num_extreme)

    # Split into train/val subsets
    val_normal_ds = normal_ds.take(n_val_normal)
    val_extreme_ds = extreme_ds.take(n_val_extreme)
    train_normal_ds = normal_ds.skip(n_val_normal)
    train_extreme_ds = extreme_ds.skip(n_val_extreme)

    # Build balanced datasets
    train_ds = make_balanced_dataset(train_normal_ds, train_extreme_ds, batch_size, extremes_per_batch_ratio=0.25)
    val_ds = make_balanced_dataset(val_normal_ds, val_extreme_ds, batch_size, extremes_per_batch_ratio=0.25)

    #del train_normal_ds,train_extreme_ds,val_normal_ds,val_extreme_ds
    return train_ds,val_ds

def model_run():
    # Function to test UNet architecture on HCW data
 
    # Arguments
    run_name = sys.argv[1]

    # Set GPU usage
    visible_devices = tf.config.list_physical_devices('GPU')
    n_visible_devices = len(visible_devices)
    print('GPUs:', visible_devices)
    if n_visible_devices > 0:
        for device in visible_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f'We have {n_visible_devices} GPUs')
        #if n_visible_devices > 1:
        #    visible_devices = visible_devices[0]
        #    print('Check to ensure we want to use more than 1 GPU; setting to 1 for now')
    else:
        print('No GPU available, passing for now')
        #raise OSError('No GPU available')
    
    # SLURM configuration: TF options, Set cluster resolver and strategy
    #resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
    #strategy = tf.distribute.MirroredStrategy(cluster_resolver=resolver)
    #strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=resolver)
    
    #options = tf.data.Options()
    #options.experimental_distribute.auto_shard_policy = (
    #    tf.data.experimental.AutoShardPolicy.DATA
    #)

    #num_workers = strategy.num_replicas_in_sync
    #worker_id = strategy.cluster_resolver.task_id
    #print('Number of workers/worker id:', num_workers, worker_id)

    # Set directory structure
    base_dir = ''
    hist_dir = os.path.join(base_dir,'hist/WRF-Monthly') 
    fut_dir = os.path.join(base_dir,'fut/WRF-Monthly') 
    
    hist_tfrecord_dir = os.path.join(hist_dir,'tfrecords')
    fut_tfrecord_dir = os.path.join(fut_dir,'tfrecords')
    
    train_dir = os.path.join(base_dir,'ML_Data/train/')
    #train_dir = os.path.join(base_dir,'ML_Data/train_4year')
    test_dir = os.path.join(base_dir,'ML_Data/test/')
    out_dir = os.path.join(base_dir,'ML/')
    
    print('Directories: ', base_dir, train_dir, test_dir)
    print('')
    print('')

    # Create TF datasets for historical and future
    print('Starting preprocessing routine...')
    preprocess_and_write_tfrecord(hist_dir,hist_tfrecord_dir)
    preprocess_and_write_tfrecord(fut_dir,fut_tfrecord_dir,epoch='fut')
    print('Finished writing to TF records...')
    print('')
    print('')
    #preprocess_and_write_tfrecord(hist_dir,tfrecord_dir)
    #sys.exit(0)

    # Read in all inputs for each climate state
    # Historical data
    os.chdir(train_dir)
    print(f'Currently in {os.getcwd()}')

    # Load model dataset
    print('Loading dataset...')
    norm_ds = load_dataset(train_dir, 'norm', batch_size=8) # Scale this by n GPUs used. I.e if using 2 GPUs with 64 GB memory for full dataset, ~32 GB per GPU, so divide batch_size by 2
    ext_ds = load_dataset(train_dir, 'ext', batch_size=8)

    norm_ds = norm_ds.unbatch()
    ext_ds = ext_ds.unbatch()

    global_batch = 8
    #per_replica_batch = global_batch // num_workers
    #print('Per replica bath size:',per_replica_batch)

    print('Making balanced dataset...')
    ds_train =  make_balanced_dataset(
        norm_ds,
        ext_ds,
        batch_size=4,
        extremes_per_batch_ratio=0.25,
        shuffle_size=256,
        num_batches_per_epoch=100,
        )
    del norm_ds,ext_ds

    #ds_train, ds_val = get_train_val_balanced_datasets(norm_ds,
    #                                                   ext_ds,
    #                                                   val_fraction=0.05,
    #                                                   batch_size=global_batch,
    #                                                   extremes_per_batch=2,
    #                                                   shuffle_buffer=128)
    #ds_train = ds_train.with_options(options)
    #ds_val   = ds_val.with_options(options)
        
    print('Training dataset:')
    print(ds_train)
    for x,y in ds_train.take(1):
        print('X shape:',x.shape)
        print('Y shape:',y.shape)
    print('----------------------') 

    ds_test = load_dataset(test_dir, 'data', batch_size=8)
    print('Testing dataset:')
    print(ds_test)
    print('----------------------')
    # 4 A100 GPUs per node
    # Model setup

    # Encoder parameters
    print('Preparing model...')
    #sample_input, _ = next(iter(ds_train))
    #X_shape = sample_input.shape[1:]
    X_shape = (512,512,12)

    #X_shape = hist_inputs.shape
    #X = ds_train
    filters = [32,64,128]
    kernel_size = 3
    stack_down = 2
    activation_down = 'ReLU'
    pool_opt = 'max'
    name = 'unet_hcw'
    
    # Decoder parameters
    stack_up = 2
    activation_up = 'ReLU'
    unpool_opt = 'bilinear'
    concat = True
        
    #with strategy.scope():
       
        #ds_train = ds_train.shard(num_workers, worker_id)
        #ds_val   = ds_val.shard(num_workers, worker_id)

        #train_ds = train_ds.shuffle(512).batch(per_replica_batch).prefetch(tf.data.AUTOTUNE)
        #val_ds   = val_ds.batch(per_replica_batch).prefetch(tf.data.AUTOTUNE)

        #train_dist = strategy.experimental_distribute_datasets_from_function(
        #        lambda input_context: ds_train.shard(
        #            input_context.num_input_pipelines,
        #            input_context.input_pipeline_id
        #        ).batch(per_replica_batch)
        #    )
        
        #val_dist = strategy.experimental_distribute_datasets_from_function(
        #        lambda input_context: ds_val.shard(
        #            input_context.num_input_pipelines,
        #            input_context.input_pipeline_id
        #        ).batch(per_replica_batch)
        #    )

    model = models.unet_3plus_2d((X_shape), # Set tensor shape
         n_labels=1,
         l1=1e-4,
         l2=1e-4,
         filter_num_down=filters,
         filter_num_skip='auto',
         filter_num_aggregate='auto',
         stack_num_down=stack_down,
         stack_num_up=stack_up,
         activation=activation_up,
         output_activation=None,
         batch_norm=True,
         pool=pool_opt,
         unpool=unpool_opt,
         name=name)

    model.compile(loss=WeightedLogMSE(alpha=5.0,beta=2.0,threshold=10.0),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))
    print(model.summary())

        
    # Instantiate callbacks
    callback_list = []
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=100)
    nan_callback = tf.keras.callbacks.TerminateOnNaN()
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'/pscratch/sd/n/nee2000/WRF-Prod/ML/model_checkpoint_{run_name}.keras',monitor='val_loss',save_freq=10)

    callback_list.extend([es_callback,nan_callback,checkpoint_callback])
        
    # Fit model
    #ds_train = ds_train.values
    #ds_val = ds_val.values
    for x,y in ds_train.take(1):
        print(x,y)
        print(x.shape,y.shape)
    print(ds_train)

    #trained = model.fit(ds_train,epochs=20,steps_per_epoch=100,validation_steps=10,callbacks=callback_list)
    trained = model.fit(ds_train,epochs=3000,steps_per_epoch=200,callbacks=callback_list)
    #trained = model.fit(train_dist,validation_data=val_dist,epochs=10,steps_per_epoch=100,validation_steps=10,callbacks=callback_list)

    #model.save(f'{run_name}_intermediate.keras')
    #print('')
    #print(f'Finished epoch #{epoch}...')

    model.save(f'{run_name}.keras')

    # Generate predictions
    pred = model.predict(ds_test)

    # Print summary statistics
    print('EDA...')
    print('Predictions shape is: ',pred.shape)
    print('Predictions mean is ', np.mean(pred))
    print('Predictions STD is ', np.std(pred))

    print('10th percentile of predictions is ', np.percentile(pred, q=10))
    print('25th percentile of predictions is ', np.percentile(pred, q=25))
    print('50th percentile of predictions is ', np.percentile(pred, q=50))
    print('75th percentile of predictions is ', np.percentile(pred, q=75))
    print('90th percentile of predictions is ', np.percentile(pred, q=90))
    print('95th percentile of predictions is ', np.percentile(pred, q=95))
    print('99th percentile of predictions is ', np.percentile(pred, q=99))
    print('99.9th percentile of predictions is ', np.percentile(pred, q=99.9))
    print('99.99th percentile of predictions is ', np.percentile(pred, q=99.99))
    print('Maximum of predictions is ', np.max(pred))
    print('-------------------------------------------------------')

    # Save results
    os.chdir(out_dir)
    with open(f'history_{run_name}.pkl','wb') as f:
        pickle.dump(trained.history,f)
    with open(f'predictions_{run_name}.pkl','wb') as f:
        pickle.dump(pred,f)
    print('Predictions saved in ',os.getcwd())


def unet_test():
    # Function to test UNet architecture on HCW data
   
    # To do to get this damn thing running:
    # Install keras unet collections - Done
    # Fill in code
    # Test loading data
    # Test running model with small subset of data
    # Do the damn thing

    # Set directory structure
    base_dir = ''
    hist_dir = os.path.join(base_dir,'hist/WRF-Monthly') 
    fut_dir = os.path.join(base_dir,'fut/WRF-Monthly') 
    hist_tfrecord_dir = os.path.join(hist_dir,'tfrecords')
    fut_tfrecord_dir = os.path.join(hist_dir,'tfrecords')
    
    # Create TF datasets for historical and future
    preprocess_and_write_tfrecord(hist_dir,hist_tfrecord_dir)
    preprocess_and_write_tfrecord(fut_dir,fut_tfrecord_dir)
    print('Directories: ', base_dir, hist_dir, fut_dir)

    # Read in all inputs for each climate state
    # Historical data
    os.chdir(hist_dir)
    print(f'Currently in {os.getcwd()}')

    cape_files = glob.glob('CAPE*.nc')
    cin_files = glob.glob('CIN*.nc')
    srh_files = glob.glob('SRH*.nc')

    apcp_files = glob.glob('APCP*.nc')
    mslp_files = glob.glob('MSLP*.nc')
    pw_files = glob.glob('PW*.nc')
    th_files = glob.glob('TH*.nc')
    wspd_files = glob.glob('WSPD*.nc')
    
    q2_files = glob.glob('Q2*.nc')
    t2_files = glob.glob('T2*.nc')
    u10_files = glob.glob('U10*.nc')
    v10_files = glob.glob('V10*.nc')

    cape_files = sorted(cape_files, key=lambda x: x[-10:-3])
    cin_files = sorted(cin_files, key=lambda x: x[-10:-3])
    srh_files = sorted(srh_files, key=lambda x: x[-10:-3])
    
    apcp_files = sorted(apcp_files, key=lambda x: x[-10:-3])
    mslp_files = sorted(mslp_files, key=lambda x: x[-10:-3])
    pw_files = sorted(pw_files, key=lambda x: x[-10:-3])
    th_files = sorted(th_files, key=lambda x: x[-10:-3])
    wspd_files = sorted(wspd_files, key=lambda x: x[-10:-3])

    q2_files = sorted(q2_files, key=lambda x: x[-10:-3])
    t2_files = sorted(t2_files, key=lambda x: x[-10:-3])
    u10_files = sorted(u10_files, key=lambda x: x[-10:-3])
    v10_files = sorted(v10_files, key=lambda x: x[-10:-3])

    #cape = xr.open_mfdataset([file for file in cape_files], concat_dim='time', combine='nested')
    #cin = xr.open_mfdataset([file for file in cin_files], concat_dim='time', combine='nested')
    #srh = xr.open_mfdataset([file for file in srh_files], concat_dim='time', combine='nested')

    #apcp = xr.open_mfdataset([file for file in apcp_files], concat_dim='time', combine='nested')
    #mslp = xr.open_mfdataset([file for file in mslp_files], concat_dim='time', combine='nested')
    #pw = xr.open_mfdataset([file for file in pw_files], concat_dim='time', combine='nested')
    #th = xr.open_mfdataset([file for file in th_files], concat_dim='time', combine='nested')
    #wspd = xr.open_mfdataset([file for file in wpsd_files], concat_dim='time', combine='nested')
    
    #q2 = xr.open_mfdataset([file for file in q2_files], concat_dim='time', combine='nested')
    #t2 = xr.open_mfdataset([file for file in t2_files], concat_dim='time', combine='nested')
    #u10 = xr.open_mfdataset([file for file in u10_files], concat_dim='time', combine='nested')
    #v10 = xr.open_mfdataset([file for file in v10_files], concat_dim='time', combine='nested')
    
    cape = xr.open_dataset(cape_files[0])
    cin = xr.open_dataset(cin_files[0])
    srh = xr.open_dataset(srh_files[0])
    
    apcp = xr.open_dataset(apcp_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'})
    mslp = xr.open_dataset(mslp_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'}).drop('Time')
    pw = xr.open_dataset(pw_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'}).drop('Time')
    th = xr.open_dataset(th_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'})

    wspd_dir = xr.open_dataset(wspd_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'}).drop('Time').drop_vars('level')
    wspd = wspd_dir.drop_sel(wspd_wdir='wdir').squeeze('wspd_wdir').rename_vars({'uvmet_wspd_wdir_interp':'wspd'}).drop(labels='wspd_wdir')
    wdir = wspd_dir.drop_sel(wspd_wdir='wspd').squeeze('wspd_wdir').rename_vars({'uvmet_wspd_wdir_interp':'wdir'}).drop(labels='wspd_wdir')

    q2 = xr.open_dataset(q2_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'})
    t2 = xr.open_dataset(t2_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'})
    u10 = xr.open_dataset(u10_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'})
    v10 = xr.open_dataset(v10_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'})

    # Set proper time indices
    # Assign range to CAPE/CIN/SRH
    time_range = pd.date_range('1991-01-01', periods = len(cape.time), freq='3H')
    cape['time'] = time_range
    cin['time'] = time_range
    srh['time'] = time_range

    # Sort remaining variables by datetime
    apcp = apcp.sortby('time')
    mslp = mslp.sortby('time')
    pw = pw.sortby('time')
    th = th.sortby('time')
    
    q2 = q2.sortby('time')
    t2 = t2.sortby('time')
    u10 = u10.sortby('time')
    v10 = v10.sortby('time')

    apcp['time'] = time_range
    mslp['time'] = time_range
    pw['time'] = time_range
    th['time'] = time_range

    q2['time'] = time_range
    t2['time'] = time_range
    u10['time'] = time_range
    v10['time'] = time_range

    # Sort WSPD and assign
    wspd = wspd.sortby('time')
    wdir = wdir.sortby('time')

    wspd['time'] = time_range
    wdir['time'] = time_range

    data_vars = [cape,cin,srh,apcp,mslp,pw,th,wspd,wdir,q2,t2,u10,v10]
    #for var in data_vars:
        #print(var)
        #print(var.nbytes / (1024 * 1024 * 1024))
        #print(var.nbytes / (1024 **3))

    #hist_inputs = xr.concat([cape,cin,srh,apcp,mslp,pw,wspd,wdir,q2,t2,u10,v10],dim='feature')
    hist_inputs = xr.merge([cape,cin,srh,apcp,mslp,pw,wspd,wdir,q2,t2,u10,v10])
    #print('Historical inputs: ', hist_inputs)
    #ds_size = hist_inputs.nbytes
    #ds_gb = ds_size / (1024 **3)
    #print('DS size in GB: ', ds_gb, ' GB')

    # Future data
    os.chdir(fut_dir)
    cape_files = glob.glob('CAPE*.nc')
    cin_files = glob.glob('CIN*.nc')
    srh_files = glob.glob('SRH*.nc')

    apcp_files = glob.glob('APCP*.nc')
    mslp_files = glob.glob('MSLP*.nc')
    pw_files = glob.glob('PW*.nc')
    th_files = glob.glob('TH*.nc')
    wspd_files = glob.glob('WSPD*.nc')
    
    q2_files = glob.glob('Q2*.nc')
    t2_files = glob.glob('T2*.nc')
    u10_files = glob.glob('U10*.nc')
    v10_files = glob.glob('V10*.nc')
    
    cape_files = sorted(cape_files, key=lambda x: x[-10:-3])
    cin_files = sorted(cin_files, key=lambda x: x[-10:-3])
    srh_files = sorted(srh_files, key=lambda x: x[-10:-3])
    
    apcp_files = sorted(apcp_files, key=lambda x: x[-10:-3])
    mslp_files = sorted(mslp_files, key=lambda x: x[-10:-3])
    pw_files = sorted(pw_files, key=lambda x: x[-10:-3])
    th_files = sorted(th_files, key=lambda x: x[-10:-3])
    wspd_files = sorted(wspd_files, key=lambda x: x[-10:-3])

    q2_files = sorted(q2_files, key=lambda x: x[-10:-3])
    t2_files = sorted(t2_files, key=lambda x: x[-10:-3])
    u10_files = sorted(u10_files, key=lambda x: x[-10:-3])
    v10_files = sorted(v10_files, key=lambda x: x[-10:-3])

    #cape = xr.open_mfdataset([file for file in cape_files], concat_dim='time', combine='nested')
    #cin = xr.open_mfdataset([file for file in cin_files], concat_dim='time', combine='nested')
    #srh = xr.open_mfdataset([file for file in srh_files], concat_dim='time', combine='nested')

    #apcp = xr.open_mfdataset([file for file in apcp_files], concat_dim='time', combine='nested')
    #mslp = xr.open_mfdataset([file for file in mslp_files], concat_dim='time', combine='nested')
    #pw = xr.open_mfdataset([file for file in pw_files], concat_dim='time', combine='nested')
    #th = xr.open_mfdataset([file for file in th_files], concat_dim='time', combine='nested')
    #wspd = xr.open_mfdataset([file for file in wpsd_files], concat_dim='time', combine='nested')
    
    #q2 = xr.open_mfdataset([file for file in q2_files], concat_dim='time', combine='nested')
    #t2 = xr.open_mfdataset([file for file in t2_files], concat_dim='time', combine='nested')
    #u10 = xr.open_mfdataset([file for file in u10_files], concat_dim='time', combine='nested')
    #v10 = xr.open_mfdataset([file for file in v10_files], concat_dim='time', combine='nested')
    
    cape = xr.open_dataset(cape_files[0])
    cin = xr.open_dataset(cin_files[0])
    srh = xr.open_dataset(srh_files[0])
    
    apcp = xr.open_dataset(apcp_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'})
    mslp = xr.open_dataset(mslp_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'}).drop('Time')
    pw = xr.open_dataset(pw_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'}).drop('Time')
    th = xr.open_dataset(th_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'})

    wspd_dir = xr.open_dataset(wspd_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'}).drop('Time').drop_vars('level')
    wspd = wspd_dir.drop_sel(wspd_wdir='wdir').squeeze('wspd_wdir').rename_vars({'uvmet_wspd_wdir_interp':'wspd'}).drop(labels='wspd_wdir')
    wdir = wspd_dir.drop_sel(wspd_wdir='wspd').squeeze('wspd_wdir').rename_vars({'uvmet_wspd_wdir_interp':'wdir'}).drop(labels='wspd_wdir')

    q2 = xr.open_dataset(q2_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'})
    t2 = xr.open_dataset(t2_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'})
    u10 = xr.open_dataset(u10_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'})
    v10 = xr.open_dataset(v10_files[0]).rename({'XLAT':'lat','XLONG':'lon','XTIME':'time','south_north':'x','west_east':'y'})
    
    # Set proper time indices
    # Assign range to CAPE/CIN/SRH
    time_range = pd.date_range('1991-01-01', periods = len(cape.time), freq='3H')
    cape['time'] = time_range
    cin['time'] = time_range
    srh['time'] = time_range

    # Sort remaining variables by datetime
    apcp = apcp.sortby('time')
    mslp = mslp.sortby('time')
    pw = pw.sortby('time')
    th = th.sortby('time')
    
    q2 = q2.sortby('time')
    t2 = t2.sortby('time')
    u10 = u10.sortby('time')
    v10 = v10.sortby('time')

    apcp['time'] = time_range
    mslp['time'] = time_range
    pw['time'] = time_range
    th['time'] = time_range

    q2['time'] = time_range
    t2['time'] = time_range
    u10['time'] = time_range
    v10['time'] = time_range

    # Sort WSPD and assign
    wspd = wspd.sortby('time')
    wdir = wdir.sortby('time')

    wspd['time'] = time_range
    wdir['time'] = time_range
    
    #fut_inputs = xr.concat([cape,cin,srh,apcp,mslp,pw,th,wspd,q2,t2,u10,v10],dim='feature')
    fut_inputs = xr.merge([cape,cin,srh,apcp,mslp,pw,wspd,wdir,q2,t2,u10,v10])
    ds_size = fut_inputs.nbytes
    ds_gb = ds_size / (1024 **3)
    print('DS size in GB: ', ds_gb, ' GB')

    # Clear individual variables from memory
    del cape,cin,srh,apcp,mslp,pw,wspd,wdir,wspd_dir,q2,t2,u10,v10

    # Read in all labels (UVV) for each climate state
    os.chdir(hist_dir)
    w_files = glob.glob('W_UP_MAX*.nc')
    w_files = sorted(w_files, key=lambda x: x[-10:-3])
    #hist_labels = xr.open_mfdataset([file for file in w_files], concat_dim='time', combine='nested')
    hist_labels = xr.open_dataset(w_files[0])

    os.chdir(fut_dir)
    w_files = glob.glob('W_UP_MAX*.nc')
    w_files = sorted(w_files, key=lambda x: x[-10:-3])
    #fut_labels = xr.open_mfdataset([file for file in w_files], concat_dim='time', combine='nested')
    fut_labels = xr.open_dataset(w_files[0])

    # Preprocess - Split, normalize, convert to numpy, etc.
    # Will want to train on historical, test on future

    # Convert to xarray DA
    hist_labels = hist_labels.transpose('x','y','time')
    fut_labels = fut_labels.transpose('x','y','time')

    # Normalize wrt historical inputs
    print('Normalizing inputs...')
    hist_inputs = hist_inputs.to_array(dim='feature').transpose('x','y','time','feature').to_numpy()
    fut_inputs = fut_inputs.to_array(dim='feature').transpose('x','y','time','feature').to_numpy()

    hist_labels = hist_labels.to_array().squeeze('variable').to_numpy()
    fut_labels = fut_labels.to_array().squeeze('variable').to_numpy()

    hist_inputs_in = (hist_inputs - np.min(hist_inputs)) / (np.max(hist_inputs) - np.min(hist_inputs))
    fut_inputs_in = (fut_inputs - np.min(hist_inputs)) / (np.max(hist_inputs) - np.min(hist_inputs))

    # Model setup
    print(hist_inputs_in.shape)
    print(hist_labels.shape)
    hist_inputs_tf = tf.convert_to_tensor(hist_inputs_in, dtype=tf.float16)
    fut_inputs_tf = tf.convert_to_tensor(fut_inputs_in, dtype=tf.float16)

    hist_labels_tf = tf.convert_to_tensor(hist_labels, dtype=tf.float16)
    fut_labels_tf = tf.convert_to_tensor(fut_labels, dtype=tf.float16)

    ds_train = tf.data.Dataset.from_tensor_slices((hist_inputs_tf,hist_labels_tf))
    ds_test = tf.data.Dataset.from_tensor_slices((fut_inputs_tf,fut_labels_tf))

    # Encoder parameters
    print('Preparing model...')
    X_shape = hist_inputs.shape
    #X = ds_train
    filters = [32,64,128]
    kernel_size = 3
    stack_down = 2
    activation_down = 'ReLU'
    pool_opt = 'max'
    name = 'unet_hcw'
    
    # Decoder parameters
    stack_up = 2
    activation_up = 'ReLU'
    unpool_opt = 'bilinear'
    concat = True

    # Create distributed strategy
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = models.unet_3plus_2d((X_shape), # Set tensor shape
             n_labels=1,
             l1=1e-4,
             l2=1e-4,
             filter_num_down=filters,
             filter_num_skip='auto',
             filter_num_aggregate='auto',
             stack_num_down=stack_down,
             stack_num_up=stack_up,
             activation=activation_up,
             output_activation=None,
             batch_norm=True,
             pool=pool_opt,
             unpool=unpool_opt,
             name=name)

        #model.compile(loss=tf.keras.losses.MeanSquaredError(),
        #          optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))
        model.compile(loss=weighted_mse,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))

    # Fit model
    trained = model.fit(ds_train)

    # Generate predictions
    preds = model.predict(ds_test)

    # Save results
    with open('predictions_out.pkl','wb') as f:
        pickle.dump(preds)

def make_balanced_dataset_OLD(normal_ds, extreme_ds, 
                          batch_size=32, extremes_per_batch=6, 
                          buffer_size=1000):
    normal_per_batch = batch_size - extremes_per_batch
    
    # Shuffle + repeat so we can zip infinitely
    #normal_ds  = normal_ds.shuffle(buffer_size).repeat().batch(normal_per_batch)
    #extreme_ds = extreme_ds.shuffle(buffer_size//4).repeat().batch(extremes_per_batch)
    
    # Zip the two datasets together and merge each batch
    #ds = tf.data.Dataset.zip((normal_ds, extreme_ds))

    #def merge_batches(norm_batch, ext_batch):
    #    x = tf.concat([norm_batch[0], ext_batch[0]], axis=0)
    #    y = tf.concat([norm_batch[1], ext_batch[1]], axis=0)
    #    return x, y
    
    #ds = ds.map(merge_batches, num_parallel_calls=tf.data.AUTOTUNE)
    #normal_ds = normal_ds.repeat()
    #extreme_ds = extreme_ds.repeat()
    
    # Shuffle
    normal_ds  = normal_ds.shuffle(buffer_size).batch(normal_per_batch)
    extreme_ds = extreme_ds.shuffle(buffer_size).batch(extremes_per_batch)

    # Compute weights
    normal_weight = (batch_size - extremes_per_batch) / batch_size
    extreme_weight = extremes_per_batch / batch_size

    # Sample from each stream according to weights
    ds = tf.data.Dataset.sample_from_datasets(
            [extreme_ds,normal_ds],
            weights=[extreme_weight,normal_weight]
    )

    ds = ds.batch(batch_size,drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def train_val():
    ds_train, ds_val = get_train_val_balanced_datasets(
            norm_ds,
            ext_ds,
            val_fraction=0.1,
            batch_size=8,
            extremes_per_batch=2)
    
    print('Making balanced dataset...')
   
    print('Training dataset:')
    print(ds_train)
    for x,y in ds_train.take(1):
        print('X shape:',x.shape)
        print('Y shape:',y.shape)
    print('----------------------')
    
    print('Validation dataset:')
    print(ds_val)
    print('----------------------')


    # Training dataset
    batch_size=8
    ds_train = (
        ds_train
        .shuffle(buffer_size=2048, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Validation dataset
    ds_val = (
        ds_val
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    batch_size = 8
    num_epochs = 10
    for epoch in range(num_epochs):
        ds_train, ds_val = get_train_val_balanced_datasets(
                norm_ds,
                ext_ds,
                val_fraction=0.1,
                batch_size=4,
                extremes_per_batch=1,
                shuffle_buffer=2048)

        trained = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=1,  # train for one epoch per loop
            steps_per_epoch=100,
            verbose=1
        )
