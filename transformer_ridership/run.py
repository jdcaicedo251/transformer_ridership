import os
import argparse
import pandas as pd
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from data import tf_data
from transformer_model_V3 import Transformer, MinMax


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "-l", "--layers", action="store",
        help="number of layers")
    parser.add_argument(
        "-d", "--d_model", action="store",
        help="d model")
    parser.add_argument(
        "-h", "--heads", action="store",
        help="number of heads")
    parser.add_argument(
        "-dff", "--dff", action="store",
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
        "-sz", "--size", action="store",
        help="max size of transactions")
    parser.add_argument(
        "-st", "--num_stations", action="store",
        help="number of stations")
    parser.add_argument(
        "-td", "--train_date", action="store",
        help="last day of training")
    parser.add_argument(
        "-k", "--key_dim", action="store",
        help="key dimension")

    args = parser.parse_args()

    num_layers = args.layers if args.layers else 4
    d_model = args.d_model if args.d_model else 10
    dff = args.dff if args.dff else 2048
    num_heads = args.heads if args.heads else 8
    dropout_rate = args.drop_rate if args.drop_rate else 0.2
    transactions_path = args.path if args.path else '../data/clean_transactions.csv'
    stations_path = args.stations if args.stations else '../data/clean_stations_database_v2.csv'
    aggregation = args.aggregation if args.aggregation else "15-mins"
    max_stations = args.num_stations if args.num_stations else None
    max_transactions = args.size if args.size else None
    train_date = args.train_date if args.train_date else '2018-08-01'
    key_dim = args.key_dim if args.key_dim else 64


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

    #Standard normalization
    norm = tf.keras.layers.Normalization(axis = None, mean = 188.4318877714359, variance = 120971.63484231419)
    # unnorm = tf.keras.layers.Normalization(axis = None, invert = True)
    # norm.adapt(train_features[:,0,:])
    # unnorm.adapt(train_features[:,0,:])
    norm = MinMax(

    transformer_norm = Transformer(
        normalizer = norm,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        key_dim = key_dim,
        dff=dff,
        dropout_rate=dropout_rate)

    # print("Transformer result shape: {}".format(transformer_norm([features[:1], time_embeddings[:1], spatial_embeddings[:1]]).shape))
    # print("Training Label Shape: {}".format(norm(labels[:1]).shape))

    # print(transformer_norm.summary())
    # print(transformer_norm.get_metrics_result())
    # print(transformer_norm.non_trainable_weights)

    #Tranining Parameters
    loss_fn = tf.losses.MeanSquaredError()
    optimizer = tf.optimizers.Adam(0.001)
    accuracy_fn = [tf.metrics.MeanAbsoluteError()]
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                                        patience=5,
                                                        mode='min')

    checkpoint_path = "training_v2/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    #Training
    transformer_norm.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])
    transformer_norm.fit(
        x = [train_features, train_time_embeddings, train_spatial_embeddings],
        y = norm(train_labels),
        epochs=100,
        callbacks=[early_stopping, cp_callback],
        batch_size=64)

    transformer_norm.save('multivariate_transformer_v2')

    # transformer_norm = tf.keras.models.load_model('multivariate_transformer')

    #Running Results
    print("")
    print("Running Predictions")
    test_features = test_data['features']
    test_time_embeddings = test_data['time_embeddings']
    test_spatial_embeddings = test_data['spatial_embeddings']

    train_results = transformer_norm.predict([train_features, train_time_embeddings, train_spatial_embeddings], batch_size = 64)
    # train_results = unnorm(train_results)
    test_results = transformer_norm.predict([test_features, test_time_embeddings, test_spatial_embeddings], batch_size = 64)
    # test_results = unnorm(test_results)
    list_stations = metadata['list_stations']
    train_date_index = metadata['train_date_index']
    test_date_index = metadata['test_date_index']

    df_train_results = pd.DataFrame(train_results, columns = list_stations, index = train_date_index)
    df_test_results = pd.DataFrame(test_results, columns = list_stations, index = test_date_index)
    df_results = pd.concat((df_train_results,df_test_results), axis = 0 )
    df_results.to_parquet('transformers_results_v2.parquet')
