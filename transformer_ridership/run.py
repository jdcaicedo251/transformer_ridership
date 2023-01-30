import os
import argparse
import pandas as pd
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from data import tf_data
from transformer_model import Transformer, MinMax, StandardNormlaization

def run_predictions(model, train_data, test_data, metadata, normalizer, clousures):

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

    if clousures:
        train_status = train_data['status']
        test_status = test_data['status']
    else:
        trian_status = tf.ones_like(train_labels)
        test_status = tf.ones_like(test_labels)

    train_inputs = [train_features, train_time_embeddings,
                        train_spatial_embeddings, train_status]
    test_inputs = [test_features, test_time_embeddings,
                       test_spatial_embeddings, test_status]

    train_results = model.predict(train_inputs, batch_size = 64)
    train_results = normalizer(train_results, reverse = True)
    df_train_results = pd.DataFrame(train_results, columns = list_stations, index = train_date_index)

    test_results = model.predict(test_inputs, batch_size = 64)
    test_results = normalizer(test_results, reverse = True)
    df_test_results = pd.DataFrame(test_results, columns = list_stations, index = test_date_index)

    df = pd.concat((df_train_results,df_test_results), axis = 0 )
    df = df.round(decimals = 0)
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)

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
        "-c", "--clousures", action='store_true',
        help="To explicitly account for station clousures")


    args = parser.parse_args()

    num_layers = args.layers if args.layers else 4
    d_model = args.d_model if args.d_model else 11
    dff = args.dff if args.dff else 2048
    num_heads = args.heads if args.heads else 8
    dropout_rate = args.drop_rate if args.drop_rate else 0.2
    transactions_path = args.path if args.path else '../data/transactions.parquet'
    stations_path = args.stations if args.stations else '../data/stations_DB.parquet'
    aggregation = args.aggregation if args.aggregation else "15-mins"
    max_stations = args.num_stations if args.num_stations else None
    max_transactions = args.size if args.size else None
    train_date = args.train_date if args.train_date else '2018-08-01'
    key_dim = args.key_dim if args.key_dim else 64
    out_name = args.out_name if args.out_name else "default"
    normalizer = args.normalizer if args.normalizer else "minmax"
    activation = args.activation if args.activation else "relu"
    attention_axes = (1) if args.attention else (1,2)
    clousures = args.clousures if args.clousures else False

    train_data, test_data, metadata = tf_data(
        transactions_path,
        stations_path,
        aggregation,
        train_date,
        max_transactions,
        max_stations)

    train_features = train_data['features']
    train_time_embeddings = train_data['time_embeddings']
    train_spatial_embeddings = train_data['spatial_embeddings']
    train_labels = train_data['labels']

    if clousures:
        train_status = train_data['status']
    else:
        train_status = tf.ones_like(train_data['status'])

    if normalizer == "standard":
        #Standard normalization
        # norm = tf.keras.layers.Normalization(axis = None, mean = 188.4318877714359, variance = 120971.63484231419)
        norm = StandardNormlaization(mean = 188.4318877714359, std = 347.809768181277)

    else:
        #MinMax Normalization
        norm = MinMax()
        norm.adapt(train_features)
        print (norm.max_x)
        print (norm.min_x)


    transformer = Transformer(
        normalizer = norm,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        key_dim = key_dim,
        dff=dff,
        attention_axes = attention_axes,
        activation = activation,
        dropout_rate=dropout_rate)


    #Tranining Parameters
    loss_fn = tf.losses.MeanSquaredError()
    optimizer = tf.optimizers.Adam(0.001)
    accuracy_fn = [tf.metrics.MeanAbsoluteError()]
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                                        patience=5,
                                                        mode='min')

    checkpoint_path = f"outputs/training_{out_name}/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    inputs = [train_features, train_time_embeddings,
              train_spatial_embeddings, train_status]

    #Training
    transformer.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])
    transformer.fit(
        x = inputs,
        y = norm(train_labels),
        epochs=100,
        callbacks=[early_stopping, cp_callback],
        batch_size=50)

    transformer.save(f"outputs/model_{out_name}")

    # transformer_norm = tf.keras.models.load_model('multivariate_transformer')

    #Running Results
    predictions = run_predictions(
        model = transformer,
        train_data = train_data,
        test_data = test_data,
        metadata = metadata,
        normalizer = norm,
        clousures = clousures)

    predictions.to_parquet(f'outputs/predictions_{out_name}.parquet')
