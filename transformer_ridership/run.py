import os
import argparse
import re
import pandas as pd
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from data import tf_data
from models.shared_layers import  LogTransformation, MinMax, StandardNormlaization, LogMinMax
from models.transformer import Transformer, PredictionTransformer
from models.LSTM import DeepPF
from models.CNN import CNN
from models.FNN import FNN

def string_arguments_error(model, closure_mode, normalizer):
    if model not in ['lstm', 'cnn', 'fnn', 'transformer', 'gnn']:
        raise ValueError("Invalid argument: string_argument for '--model' must be one of ['lstm', 'cnn', 'fnn', 'transformer', 'gnn']")
        
    if closure_mode not in ['mask','dummy',None]:
        raise ValueError("Invalid argument: string_argument '--closure' must be one of ['mask','dummy', None] ")
    
    if normalizer not in ['standard', 'minmax', 'log', 'logminmax']:
        raise ValueError("Invalid argument: string_argument '--closure' must be one of ['standard','minmax', 'log', 'logminmax'] ")
        

def run_predictions(model_class, train_data, test_data, metadata, normalizer, batch_size):

    print("")
    print("Running Predictions")

    list_stations = metadata['list_stations']

    #Train Data
    train_date_index = metadata['train_date_index']
    train_features = train_data['features']
    train_time_embeddings = train_data['time_embeddings']
    train_spatial_embeddings = train_data['spatial_embeddings']
    train_status = train_data['status']

    test_date_index = metadata['test_date_index']
    test_features = test_data['features']
    test_time_embeddings = test_data['time_embeddings']
    test_spatial_embeddings = test_data['spatial_embeddings']
    test_status = test_data['status']
    
    if model_class.name == 'transformer':

        train_inputs = [train_features, train_time_embeddings,
                            train_spatial_embeddings, train_status]
        test_inputs = [test_features, test_time_embeddings,
                           test_spatial_embeddings, test_status]
    else:
        train_inputs = [train_features, train_time_embeddings[:,-1], train_status[:,-1]]
        test_inputs = [test_features, test_time_embeddings[:,-1], test_status[:,-1]]

    train_results = model_class.predict(train_inputs, batch_size = batch_size)
    train_results = normalizer(train_results, reverse = True)
    df_train_results = pd.DataFrame(train_results, columns = list_stations, index = train_date_index)

    test_results = model_class.predict(test_inputs, batch_size = batch_size)
    test_results = normalizer(test_results, reverse = True)
    df_test_results = pd.DataFrame(test_results, columns = list_stations, index = test_date_index)

    df = pd.concat((df_train_results,df_test_results), axis = 0 )
    df = df.round(decimals = 0)
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "-m", "--model", action="store",
        help="Model. One of: lstm, cnn, fnn, transformer")
    parser.add_argument(
        "-c", "--closure", action='store',
        help="Closure mode. One of: mask, dummy. Default: None [no closure considered]")
    parser.add_argument(
        "-l", "--layers", action="store", type = int,
        help="number of layers")
    parser.add_argument(
        "-d", "--d_model", action="store", type = int,
        help="d model")
    parser.add_argument(
        "-h", "--heads", action="store", type = int,
        help="number of heads")
    parser.add_argument(
        "-dff", "--dff", action="store", type = int,
        help="dff")
    parser.add_argument(
        "-dr", "--drop_rate", action="store",
        help="drop rate")
    parser.add_argument(
        "-p", "--path", action="store",
        help="transactions file path")
    parser.add_argument(
        "-s", "--stations", action="store",
        help="stations file path")
    parser.add_argument(
        "-a", "--aggregation", action="store",
        help="aggregation")
    parser.add_argument(
        "-sz", "--size", action="store", type = int,
        help="max size of transactions")
    parser.add_argument(
        "-st", "--num_stations", action="store", type = int,
        help="number of stations")
    parser.add_argument(
        "-td", "--train_date", action="store",
        help="last day of training")
    parser.add_argument(
        "-k", "--key_dim", action="store", type = int,
        help="key dimension")
    parser.add_argument(
        "-o", "--out_name", action="store",
        help="name of the output files")
    parser.add_argument(
        "-n", "--normalizer", action="store",
        help="standard or minmax Normalization")
    parser.add_argument(
        "-ac", "--activation", action="store",
        help="tensorflow activation function")
    parser.add_argument(
        "-att", "--attention", action="store",
        help="attention axis - temporal(1) or temporal and spatial (2). Default Both")
    parser.add_argument(
        "-e", "--epoch", action="store", type = int,
        help="training epohcs")
    parser.add_argument(
        "-b", "--batch_size", action="store", type = int,
        help="batch size")
    parser.add_argument(
        "-ad", "--adjacency", action="store",
        help="path for adjacency matrix")

    args = parser.parse_args()
   

    model = args.model
    closure_mode = args.closure if args.closure else None
    num_layers = args.layers if args.layers else 4
    d_model = args.d_model if args.d_model else 11
    dff = args.dff if args.dff else 2048
    num_heads = args.heads if args.heads else 8
    dropout_rate = args.drop_rate if args.drop_rate else 0.2
    transactions_path = args.path if args.path else '../data/transactions.parquet'
    stations_path = args.stations if args.stations else '../data/stations_DB.parquet'
    adj_path = args.adjacency if args.adjacency else '../data/adjacency_matrix.parquet'
    aggregation = args.aggregation if args.aggregation else "15-mins"
    max_stations = args.num_stations if args.num_stations else None
    max_transactions = args.size if args.size else None
    train_date = args.train_date if args.train_date else '2018-08-01'
    key_dim = args.key_dim if args.key_dim else 64
    out_name = args.out_name if args.out_name else "default"
    normalizer = args.normalizer if args.normalizer else "minmax"
    activation = args.activation if args.activation else "linear"
    attention_axes = (1) if args.attention else (1,2)
    epochs = args.epoch if args.epoch else 100
    batch_size = args.batch_size if args.batch_size else 64
    
    #Errors
    string_arguments_error(model, closure_mode, normalizer)

    train_data, test_data, adj_matrix, metadata = tf_data(
        transactions_path,
        stations_path,
        adj_path,
        aggregation,
        train_date,
        max_transactions,
        max_stations)

    train_features = train_data['features']
    train_time_embeddings = train_data['time_embeddings']
    train_spatial_embeddings = train_data['spatial_embeddings']
    train_status = train_data['status']
    train_labels = train_data['labels']

    if normalizer == "standard":
        norm = StandardNormlaization(mean = 188.4318877714359, std = 347.809768181277)
        
    elif normalizer == 'log':
        norm = LogTransformation()
        
    elif normalizer == 'logminmax':
        norm = LogMinMax()

    else:
        norm = MinMax()
        norm.adapt(train_features)
        
    inputs = [train_features, train_time_embeddings[:,-1], train_status[:,-1]]
        
    if model == "transformer":

        model_class = Transformer(
                normalizer = norm,
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
                key_dim = key_dim,
                dff=dff,
                attention_axes = attention_axes,
                activation = activation,
                dropout_rate=dropout_rate,
                closure_mode = closure_mode,
                adj_matrix = None,
                name = model
        )
        inputs = [train_features, train_time_embeddings,
              train_spatial_embeddings, train_status ]
        
    elif model == "gnn":

        model_class = Transformer(
                normalizer = norm,
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
                key_dim = key_dim,
                dff=dff,
                attention_axes = attention_axes,
                activation = activation,
                dropout_rate=dropout_rate,
                closure_mode = closure_mode,
                adj_matrix = adj_matrix,
                name = model
        )
        inputs = [train_features, train_time_embeddings,
              train_spatial_embeddings, train_status ]
    
    
    elif model == 'lstm':
        model_class = DeepPF(
            normalizer = norm,
            closure_mode = closure_mode,
            name = model
        )
    elif model == 'cnn':
        model_class = CNN(
            normalizer = norm,
            closure_mode = closure_mode,
            name = model
        )
    else :
        model_class = FNN(
            normalizer = norm,
            closure_mode = closure_mode,
            name = model
        )
   
    #Tranining Parameters
    loss_fn = tf.losses.MeanSquaredError()
    optimizer = tf.optimizers.Adam(0.001)
    accuracy_fn = [tf.metrics.MeanAbsoluteError()]
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                                        patience=5,
                                                        mode='min')

    checkpoint_dir = f"outputs/{out_name}_{model}_checkpoints"
    checkpoint_name = "cp-{epoch:02d}.ckpt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    
    #Check if Checkpoints are available
    checkpoint_path_tmp = tf.train.latest_checkpoint(checkpoint_dir)

    if checkpoint_path_tmp is not None:
        print ("Loading From Last available checkpoint")
        model_class.load_weights(checkpoint_path_tmp)

        try: 
            initial_epoch = int(re.search(r'\d+', checkpoint_path_tmp)[0])
        except TypeError:
            print ("No epoch found in last checkpoint. Initializing in 0")
            initial_epoch = 0

    else:
        print("No checkpoint found at {}".format(checkpoint_dir))
        initial_epoch = 0
    
    
    #Training
    model_class.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])
    model_class.fit(
        x = inputs,
        y = norm(train_labels),
        epochs=epochs,
        callbacks=[early_stopping, cp_callback],
        batch_size=batch_size, 
        initial_epoch=initial_epoch)

    model_class.save(f"outputs/{out_name}_{model}_model")

    # transformer_norm = tf.keras.models.load_model('multivariate_transformer')

    #Running Results
    predictions = run_predictions(
        model_class = model_class,
        train_data = train_data,
        test_data = test_data,
        metadata = metadata,
        normalizer = norm, 
        batch_size = batch_size)

    predictions.to_parquet(f'outputs/{out_name}_{model}_predictions.parquet')
    
    if model in ["transformer", "gnn"]:
        final_model = PredictionTransformer(model_class)
        tf.saved_model.save(final_model, export_dir=f"outputs/{out_name}_{model}_model")
        
    else:
        model_class.save(f"outputs/{out_name}_{model}_model")
