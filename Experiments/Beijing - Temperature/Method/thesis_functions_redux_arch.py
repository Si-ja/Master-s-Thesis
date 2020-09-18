def wave(instances,
         duration,
         add_err,
         wave_type="cos",
         show=True,
         save_fig=False):
    """
                                         Description:
    ------------------------------------------------------------------------------------------
    This function will create data for a sin or a cos wave. Dummy data to experiment with.
    ------------------------------------------------------------------------------------------

                                      Input Parameters:
    ------------------------------------------------------------------------------------------
    > instances = amount of data points generated;
    > duration  = how long is the x-axis;
    > add_err   = addition of random noise;
    > wave_type = "cos" or "sin" waves. DEFAULT: "cos";
    > show      = show the graph of how the data looks. DEFAULT: True;
    > save_fig  = save the figure or not. DEFAULT: False;
    ------------------------------------------------------------------------------------------

                                          Outputs:
    ------------------------------------------------------------------------------------------
    > y = data points generated
    ------------------------------------------------------------------------------------------"""

    # load packages:
    import numpy as np
    import matplotlib.pyplot as plt

    # Create data for the x-axis and noise
    x = np.linspace(0, duration, instances)
    noise = np.random.normal(0, add_err, len(x))

    # Decide if sin or cos waves are built and create those data points a y-axis
    if wave_type == "cos":
        y = np.cos(x) + noise
    elif wave_type == "sin":
        y = np.sin(x) + noise
    else:
        print("You indicated a type '{}' of a wave. Currently support only exists for 'cos' or 'sin'.".format \
                  (wave_type))
        return

    # Decide if we want to show the graph for how the data was generated
    if show == True:
        f = plt.figure(figsize=(8, 6), dpi=80)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.scatter(x, y)
        plt.xlabel("Time")
        plt.ylabel(wave_type + " deviation")
        plt.title(wave_type + " wave with noise")
        plt.show()

    # Decide if the graph should be saved of the data visualized
    if save_fig == True:
        f = plt.figure(figsize=(8, 6), dpi=80)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.scatter(x, y)
        plt.xlabel("Time")
        plt.ylabel(wave_type + " deviation")
        plt.title(wave_type + " wave with noise")
        f.savefig("Wave_generated.pdf", bbox_inches='tight')
        plt.close(f)

    # Return data for future use
    return y

def gpu_activation(signal="cpu",
                   confirm=False):
    """
                                        Description:
    ----------------------------------------------------------------------------------------
    Define which component will be handeling data manipulation in the network
    ----------------------------------------------------------------------------------------

                                      Input Parameters:
    ----------------------------------------------------------------------------------------
    > signal  = Either 'gpu' or 'cpu'. DEFAULT: 'cpu';
    > confirm = Show which device was set-up. DEFAULT: False;
    ----------------------------------------------------------------------------------------

                                         Outputs:
    ----------------------------------------------------------------------------------------
    > device = specific device that has properties of either running 'cpu' or 'gpu';
    ----------------------------------------------------------------------------------------
    """
    # Import packages
    import torch

    # Prepare the device
    if signal == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if confirm == True:
        print("Currently [{}] was set up for usage".format(device))

    # Return the pre-sets for the device
    return device

def window_slider(data,
                  initial_data,
                  pred_data,
                  splitter=2 / 3,
                  prepare_log=False,
                  integrity=False,
                  save_fig=False):
    """
                                                 Description:
    ----------------------------------------------------------------------------------------------------------------
    This function will prepare the data for training and testing.

    The training data will be created with a sliding window itterator increases by 1 unit over time.
    The sliding window for the testing data will be increasing by the amount of units we need to predict.

    NB. as work is done on univariate dataset - it will be checked whether the data given has only one dimension.
    Data will be seperated into the correct format when it's feed into the LSTM network right at that moment. Current
    actions will only prepare everything in arrays and sequences, for convenience of data processing.
    ----------------------------------------------------------------------------------------------------------------

                                                Input Parameters:
    ----------------------------------------------------------------------------------------------------------------
    > data         = dataset which will be used for being seperated;
    > initial_data = amount of data points that are used for making predictions;
    > pred_data    = amount of data points to predict into the future;
    > splitter     = how much information should be dedicated to the training. DEFAULT: 2/3;
    > prepare_log  = will give a short summary on what was prepared with the data and print it. DEFAULT: False;
    > integrity    = indicate whether you want the graph for the training outputs to be visualized. DEFAULT: False;
    > save_fig     = Save the figure of what data pattern needs to be predicted. DEFAULT: False;
    ----------------------------------------------------------------------------------------------------------------

                                                      Outputs:
    ----------------------------------------------------------------------------------------------------------------
    > x_training     = training data, data;
    > y_training     = training data, labels/target values;

    > x_testing      = testing data, data;
    > y_testing      = testing data, labels/target values;
    ----------------------------------------------------------------------------------------------------------------
    """

    # Import packages
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    # Transform data if it's not in a numpy format already
    data = np.array(data)

    if data.ndim != 1:
        print("Your data seems to be of {} dimensions, when it should be only 1.".format(data.ndim))
        return

    # Prepare information that will be used for the data that we will train info on and its' targets
    data_holder = []
    targets_holder = []

    # Further build the data accumulator
    idx_start = 0  # starting point
    idx_finish = (initial_data + pred_data)  # finishing position (this should not exceed our data)
    while idx_finish <= len(data):
        data_holder.append(data[idx_start: idx_start + initial_data])
        targets_holder.append(data[idx_start + initial_data: idx_finish])

        idx_start += pred_data
        idx_finish += pred_data

    data_holder = np.array(data_holder)
    targets_holder = np.array(targets_holder)

    # Split the data for testing
    for_training = int(data_holder.shape[0] * splitter)

    x_testing = data_holder[for_training:]
    y_testing = targets_holder[for_training:]

    # NOW WE CHANGE THE SLIDING WINDOW
    # THIS WILL GIVE US MORE DATA TO TRAIN ON AND HOPEFULLY IN A MORE EFFICIENT MANNER
    # THIS DATA WILL BE FOR THE TRAINING AND WE NEED TO PARSE IT FIRST

    data_holder = []
    targets_holder = []

    for amount in range(0, int(len(data) * splitter) - initial_data - pred_data):
        data_holder.append(data[amount: initial_data + amount])
        targets_holder.append(data[amount + initial_data: amount + pred_data + initial_data])

    data_holder = np.array(data_holder)
    targets_holder = np.array(targets_holder)

    x_training = data_holder
    y_training = targets_holder

    if prepare_log == True:
        print("--------------------------------LOG-----------------------------------")
        print("\tTraining data has the following shape: \t\t{}".format(x_training.shape))
        print("\tTraining targets have the following shape: \t{}".format(y_training.shape))
        print()
        print("\tTesting data has the following shape: \t\t{}".format(x_testing.shape))
        print("\tTesting targets have the following shape: \t{}".format(y_testing.shape))
        #         print()
        #         print("First 3 instances of the training data and targets:")
        #         print(x_training[0])
        #         print(x_training[1])
        #         print(x_training[2])
        #         print("----------------------------------------------")
        #         print(y_training[0])
        #         print(y_training[1])
        #         print(y_training[2])
        #         print()
        #         print("First 3 instances of the testing data and targets:")
        #         print(x_testing[0])
        #         print(x_testing[1])
        #         print(x_testing[2])
        #         print("----------------------------------------------")
        #         print(y_testing[0])
        #         print(y_testing[1])
        #         print(y_testing[2])
        print("----------------------------------------------------------------------")

    # Transform information into tensors
    x_training = torch.tensor(x_training)
    y_training = torch.tensor(y_training)

    x_testing = torch.tensor(x_testing)
    y_testing = torch.tensor(y_testing)

    if integrity == True:
        f = plt.figure(figsize=(8, 6), dpi=80)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        # Visualize data for the testing set targets
        collector = []
        for value in range(y_testing.shape[0]):
            for number in y_testing[value].numpy():
                collector.append(number)

        plt.scatter(y=collector, x=range(0, len(collector)), c="red")
        plt.xlabel("Time", fontsize=11)
        plt.ylabel("Deviations", fontsize=11)
        plt.title("Testing targets", fontsize=11)
        plt.show()

    if save_fig == True:
        f = plt.figure(figsize=(8, 6), dpi=80)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        collector = []
        for value in range(y_testing.shape[0]):
            for number in y_testing[value].numpy():
                collector.append(number)

        plt.scatter(y=collector, x=range(0, len(collector)), c="red")
        plt.xlabel("Time", fontsize=11)
        plt.ylabel("Deviations", fontsize=11)
        plt.title("Testing targets", fontsize=11)
        f.savefig("Testing targets.pdf", bbox_inches='tight')
        plt.close(f)

    return x_training, y_training, x_testing, y_testing

def loaders(x_training,
            y_training,
            x_testing,
            y_testing,
            BATCH=10,
            SHUFFLE_TRAIN=True,
            SHUFFLE_TEST=False,
            verify=False):
    """
                                                    Description:
    -------------------------------------------------------------------------------------------------------------------
    This will prepare the data loaders, so data is loaded in batches and not all together.
    -------------------------------------------------------------------------------------------------------------------

                                                  Input Parameters:
    -------------------------------------------------------------------------------------------------------------------
    > x_training    = the tensor with the training data;
    > y_training    = the tensor with the training labels;
    > x_testing     = the tensor with the testing data;
    > y_testing     = the tensor with the testing labels;
    > BATCH         = the size of the batch that will be feed into the network. DEFAULT: 10;
    > SHUFFLE_TRAIN = indicate whether the training data during loading should be shuffled. Default: True;
    > SHUFFLE_TEST  = indicate whether the testing data during loading should be shuffled. Default: False;
    > verify        = indicate whether you want a shor log description of what sizes/shapes your loading tensors will be.
                      Default: False;
    -------------------------------------------------------------------------------------------------------------------

                                                      Outputs:
    -------------------------------------------------------------------------------------------------------------------
    > train_loaders = an object that will load information into the network for training;
    > test_loader   = an object that will load information into the network for testing;
    -------------------------------------------------------------------------------------------------------------------
    """

    # Import packages
    import torch.utils.data as data_utils

    # Prepare the train loader
    train_data = data_utils.TensorDataset(x_training, y_training)
    train_loader = data_utils.DataLoader(train_data,
                                         batch_size=BATCH,
                                         shuffle=SHUFFLE_TRAIN)

    # Prepare the test loader
    test_data = data_utils.TensorDataset(x_testing, y_testing)
    test_loader = data_utils.DataLoader(test_data,
                                        batch_size=BATCH,
                                        shuffle=SHUFFLE_TEST)

    if verify == True:
        batch = next(iter(train_loader))
        data_batch, labels_batch = batch

        print(
            "------------------------------------------------LOG-------------------------------------------------------")
        print("\t\tTraining data has the following shape per batch:\t {}".format(data_batch.shape))
        print("\t\tTraining labels have the following shape per batch:\t {}".format(labels_batch.shape))

        batch = next(iter(test_loader))
        data_batch, labels_batch = batch

        print("\t\tTesting data has the following shape per batch:\t\t {}".format(data_batch.shape))
        print("\t\tTesting labels have the following shape per batch:\t {}".format(labels_batch.shape))

        if data_batch.dim() == 1 & labels_batch.dim() == 1:
            print()
            print("\tIt seems your data is of a 2 dimensional type. Consider reshaping to a 3 dim. before training.")
        print(
            "-----------------------------------------------------------------------------------------------------------")

    return train_loader, test_loader

def baseline(true_dataset, true_labels):
    """
    Description:
    ------------------------------------------------------------------------------------------
    This function creates an artificial baseline to strive to do better than it. All the values
    to predict are equated to the means of what came before (that time step). It is a very naive
    approach, but one that can be made as a baseline to predict how well the neural network does.
    ------------------------------------------------------------------------------------------
    
    Input Parameters:
    ------------------------------------------------------------------------------------------
    > true_dataset = the data from which the predictions need to be made;
    > true_labels  = the true labels of what we want to predict;
    ------------------------------------------------------------------------------------------
    
    Output:
    ------------------------------------------------------------------------------------------
    > pred_collector = predicted data (means)
    > answ_collector = the real data to which the predictions can be compared
    ------------------------------------------------------------------------------------------

    
    """
    # Import packages
    import numpy as np
    
    predicted_from = true_dataset.shape[0] # From how many values to predict
    predicted_to   = true_labels.shape[1]  # How many values to predict
    
    pred_collector = []
    collector_temp = []
    for base in range(predicted_from):
        to_evaluate = np.mean(np.array(true_dataset[base]))
        for itter in range(predicted_to):
            collector_temp.append(to_evaluate)
        pred_collector.append(collector_temp)
        collector_temp = []
        
    collector_temp = None
    pred_collector = np.array(pred_collector) # collection of naive answers 
    
    answ_collector = []
    collector_temp = []
#     print(true_labels[0])
    for base in range(true_labels.shape[0]):
        to_evaluate = np.array(true_labels[base])
        collector_temp.append(to_evaluate)
        answ_collector.append(collector_temp)
        collector_temp = []
        
    collector_temp = []
    answ_collector = np.array(answ_collector)
            
    
    return pred_collector, answ_collector

def model_build(hidden_nn,
                data_inputs,
                data_targets,
                device,
                num_layers=1,
                dropout=0.0,
                bias=True,
                verify=False):
    """
                                        Description:
    ---------------------------------------------------------------------------------------------
    This function builds a model with which experiments can take place;
    ---------------------------------------------------------------------------------------------

                                      Input Parameters:
    ---------------------------------------------------------------------------------------------
    > hidden_nn    = how many hidden neurons will the network layers have;
    > data_input   = how many data points are feed as inputs;
    > data_targets = how many values have to be predicted (i.e. final outputs);
    > device       = to what device will data be transfered for training;
    > num_layers   = amount of layers a network will have. DEFAULT: 1;
    > dropout      = if the network has more than 1 layer, a dropout of neurons per training could
                     be indicated. DEFAULT: 0.0
    > bias         = should all layers have bias turned on or not. DEFAULT: True
    > verify       = show information indicating what kind of network was created. DEFAULT: False
    ---------------------------------------------------------------------------------------------

                                           Output:
    ---------------------------------------------------------------------------------------------
    > model = a structure of the network
    ---------------------------------------------------------------------------------------------
    """

    # Import packages
    import torch
    import torch.nn as nn
    import numpy as np

    # Make values re-assignment (due to legacy build)
    NUM_LAYERS = num_layers
    HIDDEN_NN = hidden_nn
    INITIAL_DATA = data_inputs
    PRED_DATA = data_targets
    DROPOUT = dropout
    BIAS = bias

    # Create a network
    class LSTM_NN(nn.Module):

        def __init__(self,
                     viz=False,  # Will indicate whether we should visualize the transformed tensors
                     t_pass=[]):  # Collects and holds data to visualize outputs
            super(LSTM_NN, self).__init__()
            # Several Misc items
            self.viz = viz
            self.t_pass = t_pass

            # This is an lstm layer representative
            self.lstm = nn.LSTM(input_size=1,  # How many values sequentially will be fed into it
                                hidden_size=HIDDEN_NN,  # How many neurons there are in the layer
                                num_layers=NUM_LAYERS,  # How many layers are there in the network
                                bias=BIAS,  # Will the bias be present
                                dropout=DROPOUT,  # With one layer we will always have a 0.0 dropout
                                batch_first=True,  # Initiate for batch to go as the first value
                                bidirectional=False)  # Turn off the biderectional function

            # This is the output layer representative
            self.fc = nn.Linear(in_features=HIDDEN_NN * INITIAL_DATA,  # How many cells will output their values
                                out_features=PRED_DATA,  # How many values are we predicting
                                bias=BIAS)  # Will the associated bias be counted

        def forward(self, t):
            if self.viz == False:
                # Initial input layer (0)
                batch_size, _, _ = t.size()  # Batch size will change in the end, so we need an internal tracker of it
                t = t  # (batch, initial time steps, #variables)

                # Set initial hidden and general cells states
                h0 = torch.zeros(NUM_LAYERS, batch_size, HIDDEN_NN, dtype=torch.double).to(
                    device)  # (layers, batch, neurons)
                c0 = torch.zeros(NUM_LAYERS, batch_size, HIDDEN_NN, dtype=torch.double).to(
                    device)  # (layers, batch, neurons)

                # Forward propogate LSTM
                out, (hn, cn) = self.lstm(t, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
                # print(out)  # Batch x #Inputs x #Neurons

                out = out.contiguous().view(batch_size, -1)  # Otherwise we get a matrix of  Batch x #Inputs x #Neurons
                # But this way now we get Batch x Inputs*Neurons

                # Decode the hidden state of the last time step
                out = self.fc(out)  # (Batch, Predicted Time Steps)
                return out

            elif self.viz == True:
                # This will activate if we make predictions and want to visualize them
                batch_size, _, _ = t.size()
                t = t

                h0 = torch.zeros(NUM_LAYERS, batch_size, HIDDEN_NN, dtype=torch.double).to(device)
                c0 = torch.zeros(NUM_LAYERS, batch_size, HIDDEN_NN, dtype=torch.double).to(device)

                out, (hn, cn) = self.lstm(t, (h0, c0))

                viz_copy = out.cpu().numpy()  # We need an internal temporary storage of original outputs
                # As they are far easier to work with

                # This further small function sums the outputs of cells in a neuron
                # The mutual sum is compared to what the other neurons produce
                # Comparison is done through summing of all neuron outputs (cells are summed with absolute values)
                # The summed values are divided by the total amount of outputs summed
                # All transformation become percentages that in total sum to 1 (or 100%)
                # DISCLAIMER: Softmax would produce a similar output, but variation would differ slightly
                # First pass is saved into a t_pass list holder
                for batch_entry in viz_copy:
                    neurons_collector = []
                    for neurons_entry in batch_entry.T:
#                         neurons_collector.append(np.absolute(neurons_entry).sum()) # Temporary change
                        neurons_collector.append(np.std(neurons_entry))
                    neurons_collector = np.array(neurons_collector)
                    signal_proc = neurons_collector / neurons_collector.sum()
                    self.t_pass.append(signal_proc)

                out = out.contiguous().view(batch_size, -1)
                out = self.fc(out)
                return out

        def retrieve_signal(self):
            """Return the tensor transformation visualization dataframe."""
            return np.array(self.t_pass)

        def clean_signal(self):
            """Clean the data stored from previous visualization attemtps."""
            self.t_pass = []

    model = LSTM_NN().to(device)
    model = model.double()

    if verify == True:
        print(model)

    return model

def training_init(model,
                  train_loader,
                  epochs,
                  data_input,
                  data_targets,
                  device,
                  criterion,
                  optimizer,
                  frequency=500,
                  verbose=0):
    """
                                                    Description:
    ---------------------------------------------------------------------------------------------------------
    Initialize the training of the network required
    ---------------------------------------------------------------------------------------------------------

                                                  Input Parameters:
    ---------------------------------------------------------------------------------------------------------
    > model        = the instance of the model/network we want to work with;
    > train_loader = train_loader that will be used to feed in data into the network;
    > epochs       = amount of epochs for which our network will train;
    > data_input   = how many values are used to make predictions on
    > data_targets = how many values are needed to be predicted
    > device       = what device do you want to move your information to;
    > criterion    = type of loss calculation system that you want to use;
    > optimizer    = type of an optimizer you want to use in your network;
    > frequency    = how often updates on each epoch finishing proceeding should show. DEFAULT: 500;
    > verbose      = 0 - do not show any information on the progress except which epoch is running,
                     1 - only show the epochs running and how many instances per epoch already ran,
                     2 - show epochs running and the time,
                     3 - show only the time it took to train the network,
                     DEFAULT: 0;
    ---------------------------------------------------------------------------------------------------------

                                                       Output:
    ---------------------------------------------------------------------------------------------------------
    > model = model after training, but it will update from the global environment
    ---------------------------------------------------------------------------------------------------------
    """

    # Import packages
    import time
    import numpy as np

    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torch.optim as optim
    import torch.utils.data as data_utils

    # Re-assign values to help keep track of few things
    model = model
    train_loader = train_loader
    EPOCHS = epochs
    INITIAL_DATA = data_input
    PRED_DATA = data_targets
    frequency = frequency
    verbose = verbose

    # TRAIN OUR MODEL
    model.vis = False  # Turn of the vizualization component of layers
    model.train()  # Put the model into the training state
    total_step = len(train_loader)
    start = time.time()  # Time how long it takes to execute the training
    print("Training initiated")
    for epoch in range(EPOCHS):
        if verbose != 3:
            print("EPOCH: {}".format(epoch + 1))

        for i, (data_epochs, labels_epochs) in enumerate(train_loader):
            # REPREPARE THE DATA AS WE NEED TO RESHAPE IT AT THIS POINT TO A 3D TENSOR
            data_epochs = data_epochs.reshape(-1, INITIAL_DATA, 1).to(device)  # (batch, initial time steps, #variables)
            labels_epochs = labels_epochs.reshape(-1, PRED_DATA).to(device)  # (batch, predicted time steps)

            #             TO WORK OUT THE BUGS WITH BROADCASTING
            #             print(data_epochs.shape)
            #             print(labels_epochs.shape)

            # FORWARD THE LEARING PROCESS THEN NOW
            outputs = model(data_epochs)
            #             TO WORK OUT THE BUGS WITH BROADCASTING
            #             print(outputs.shape)
            #             print(labels_epochs.shape)
            loss = criterion(outputs, labels_epochs)

            # BACKWARDS AND OPTIMIZE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose == 2 or verbose == 1:
                if (i + 1) % frequency == 0:
                    print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                          .format(epoch + 1, EPOCHS, i + 1, total_step, loss.item()))
        if verbose == 2 or verbose == 1:
            print()

    print("Training finished")
    end = time.time()
    final = end - start
    final_time = np.round(final / 60, 2)
    if verbose == 2 or verbose == 3:
        print("It took the network: {} minutes to train.".format(final_time))

def testing_init(model,
                 test_loader,
                 input_data,
                 device):
    """
                                                Description:
    -----------------------------------------------------------------------------------------------------------
    Initiate the testing of the model.
    -----------------------------------------------------------------------------------------------------------

                                              Input Parameters:
    -----------------------------------------------------------------------------------------------------------
    > model       = model that will be used for data evaluation;
    > test_loader = loader that will load our information in batches;
    > input_data  = how many values are used to make predictions on;
    > device      = what device do you want to load the information to;
    -----------------------------------------------------------------------------------------------------------

                                                   Output:
    -----------------------------------------------------------------------------------------------------------
    > final_answers = a set of data that our model produces;
    > final_data    = a set of data that we know is correct and to which our answers can be compared to;
    -----------------------------------------------------------------------------------------------------------
    """
    # Updated due to legacy connections
    INITIAL_DATA = input_data

    # Import needed packages
    import time
    import numpy as np

    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torch.optim as optim
    import torch.utils.data as data_utils

    model.viz = True
    model.clean_signal()
    model.eval()

    # Predict the data with the model we have
    with torch.no_grad():
        answers = []  # Answers that our model produces
        data_known = []  # Real Answers that we know

        for data_testing, labels_testing in (test_loader):
            data_testing = data_testing.reshape(-1, INITIAL_DATA, 1).to(device)
            labels_testing = labels_testing.to(device)

            outputs = model(data_testing)
            answers.append(outputs)
            data_known.append(labels_testing)
            
        
        PRED_DATA = labels_testing.shape[1]
        if PRED_DATA == 1:
            # Data that we predict
            final_answers = []
            for i in range(len(answers)):
                for i2 in range(len(answers[i])):
        #             for i3 in range(len(answers[i][i2])):
                    final_answers.append(answers[i][i2].cpu().numpy())
            final_answers = np.array(final_answers)

            # Data that we know to which we can compare our answers to
            final_data = []
            for i in range(len(data_known)):
                for i2 in range(len(data_known[i])):
        #             for i3 in range(len(data_known[i][i2])):
                    final_data.append(data_known[i][i2].cpu().numpy())
            final_data = np.array(final_data)

            return final_answers, final_data
        
        elif PRED_DATA > 1:
                        # Data that we predict
            final_answers = []
            for i in range(len(answers)):
                for i2 in range(len(answers[i])):
                    for i3 in range(len(answers[i][i2])):
                        final_answers.append(answers[i][i2][i3].cpu().numpy())
            final_answers = np.array(final_answers)

            # Data that we know to which we can compare our answers to
            final_data = []
            for i in range(len(data_known)):
                for i2 in range(len(data_known[i])):
                    for i3 in range(len(data_known[i][i2])):
                        final_data.append(data_known[i][i2][i3].cpu().numpy())
            final_data = np.array(final_data)

            return final_answers, final_data

def accuracy_check(predictions,
                   reality,
                   model,
                   mse=True,
                   mae=False,
                   rmse=False,
                   visualize=False,
                   save_fig=False,
                   logger=False,
                   txt_logger = False,
                   baseline = False):
    """
    Description:
    --------------------------------------------------------------------------------------------------------------
    This function will check the accuracy of predictions by the model to the reality (real data).
    --------------------------------------------------------------------------------------------------------------

    Input Parameters:
    --------------------------------------------------------------------------------------------------------------
    > predictions = a set of data that was predicted by the model (1D tensor);
    > reality     = a set of data that we know is correct (1D tensor);
    > model       = indicate which model was used. Only important if you want to save figures produced;
    > mse         = mean squared error for comparing the accuracy of the data predicted. DEFAULT: True;
    > mae         = mean absoulute error for comparing the accuracy of the data predicted. DEFAULT: False;
    > rmse        = root-mean-squared error for comparing the accuracy of the data predicted. DEFAULT: False;
    > visualize   = show the graphs with all the calculations if chosen. DEFAULT: False;
    > save_fig    = will save the figures produced to pdf for work in LaTeX. DEFAULT: False;
    > logger      = will keep the log of the errors in a txt file. DEFAULT: False;
    > txt_logger  = saves information for the errors in one line of text, for later automatic analysis. DEFAULT: False;
    > baseline    = indicate whether you want to see the accuracy if only baseline predictions are made. In that case
                    insert any value for the model, it will be ignored. Logger is deactivated for it, but the values
                    of the baseline will be saved in the similar format as for the txt_logger. DEFAULT: False;
    --------------------------------------------------------------------------------------------------------------

    Output:
    --------------------------------------------------------------------------------------------------------------
    None
    --------------------------------------------------------------------------------------------------------------
    """
    # Import packages
    import numpy as np

    # Collect all the compariosons
    if mse == True:
        from sklearn.metrics import mean_squared_error
        answer_mse = mean_squared_error(reality, predictions)

    if mae == True:
        from sklearn.metrics import mean_absolute_error
        answer_mae = mean_absolute_error(reality, predictions)

    if rmse == True:
        from sklearn.metrics import mean_squared_error
        from math import sqrt
        answer_rmse = sqrt(mean_squared_error(reality, predictions))

    # Show what is calculated
    if visualize == False:
        if mse == True:
            print("The mse score is: {}".format(np.round(answer_mse, 4)))
        if mae == True:
            print("The mae score is: {}".format(np.round(answer_mae, 4)))
        if rmse == True:
            print("The rmse score is: {}".format(np.round(answer_rmse, 4)))

    else:
        import matplotlib.pyplot as plt
        f = plt.figure(figsize=(8, 6), dpi=80)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        if mse == True:
            print("The mse score is: {}".format(np.round(answer_mse, 4)))
        if mae == True:
            print("The mae score is: {}".format(np.round(answer_mae, 4)))
        if rmse == True:
            print("The rmse score is: {}".format(np.round(answer_rmse, 4)))

        plt.plot(reality, alpha=0.5, label="Real Data", color="r")
        plt.plot(predictions, alpha=1, label="Predicted Data")

        if mse == True:
            plt.plot([], [], ' ', label="MSE error is: {}".format((np.round(answer_mse, 4))))
        if mae == True:
            plt.plot([], [], ' ', label=("MAE error is: {}".format(np.round(answer_mae, 4))))
        if rmse == True:
            plt.plot([], [], ' ', label=("RMSE error is: {}".format(np.round(answer_rmse, 4))))

        plt.xlabel("Time", fontsize=11)
        plt.ylabel("Deviation", fontsize=11)
        plt.title("Predictions and Reality", fontsize=11)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.show()

    # Decide if you want to save the outputs visualized in comparison with the errors shown
    if save_fig == True:
        if baseline == False:
            import matplotlib.pyplot as plt
            f = plt.figure(figsize=(8, 6), dpi=80)
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.plot(reality, alpha=0.5, label="Real Data", color="r")
            plt.plot(predictions, alpha=1, label="Predicted Data")
            if mse == True:
                plt.plot([], [], ' ', label="MSE error is: {}".format((np.round(answer_mse, 4))))
            if mae == True:
                plt.plot([], [], ' ', label=("MAE error is: {}".format(np.round(answer_mae, 4))))
            if rmse == True:
                plt.plot([], [], ' ', label=("RMSE error is: {}".format(np.round(answer_rmse, 4))))
            plt.xlabel("Time", fontsize=11)
            plt.ylabel("Deviation", fontsize=11)
            plt.title("Predictions and Reality", fontsize=11)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            layers_plt = model.lstm.num_layers  # How many layers were used here
            neurons_plt = model.lstm.hidden_size  # How many neurons were used here
            fname = str("Prediction and Reality _ model uses {} layers, {} neurons.pdf".format(layers_plt, neurons_plt))
            f.savefig(fname, bbox_inches='tight')
            plt.close(f)
            
        if baseline == True:
            import matplotlib.pyplot as plt
            f = plt.figure(figsize=(8, 6), dpi=80)
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.plot(reality, alpha=0.5, label="Real Data", color="r")
            plt.plot(predictions, alpha=1, label="Predicted Data")
            if mse == True:
                plt.plot([], [], ' ', label="MSE error is: {}".format((np.round(answer_mse, 4))))
            if mae == True:
                plt.plot([], [], ' ', label=("MAE error is: {}".format(np.round(answer_mae, 4))))
            if rmse == True:
                plt.plot([], [], ' ', label=("RMSE error is: {}".format(np.round(answer_rmse, 4))))
            plt.xlabel("Time", fontsize=11)
            plt.ylabel("Deviation", fontsize=11)
            plt.title("Predictions and Reality", fontsize=11)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            fname = str("Prediction and Reality _ baseline.pdf")
            f.savefig(fname, bbox_inches='tight')
            plt.close(f)
            
    if baseline == True:
        logger = False
    if logger == True:
        import datetime
        import time

        # Create a file and populate it // we can keep a long log by appending the information
        # and just indicating the real time of the log being made to not lose track of what was done
        outF = open("Full Data Log _ Thesis.txt", "a+")

        # Calculate current time
        currentDT = str(datetime.datetime.now())
        time_info = str("\n\nTime of the log made: {}\n".format(currentDT))
        outF.write(time_info)

        # Indicate for which model the calculations are done
        layers_plt = model.lstm.num_layers  # How many layers were used here
        neurons_plt = model.lstm.hidden_size  # How many neurons were used here
        outF.write("Layers: {}, Neurons: {}\n".format(layers_plt, neurons_plt))

        # Fill in the information for the metrics (errors)
        if mse == True:
            outF.write("MSE:  {}\n".format(np.round(answer_mse, 4)))
        if mae == True:
            outF.write("MAE:  {}\n".format(np.round(answer_mae, 4)))
        if rmse == True:
            outF.write("RMSE: {}\n".format(np.round(answer_rmse, 4)))

    if baseline == True:
        txt_logger = False
        outF = open("errors_analysis.txt", "a+")
        string_to_write = str()
        string_to_write = str(0) + (",") + str(0)
        string_to_write = string_to_write + (",") + str(np.round(answer_mse, 4))
        string_to_write = string_to_write + (",") + str(np.round(answer_mae, 4))
        string_to_write = string_to_write + (",") + str(np.round(answer_rmse, 4))
        string_to_write = string_to_write + (",") + ("True") + ("\n")
        outF.write(string_to_write)
        outF.close()
        
    if txt_logger == True:
        outF = open("errors_analysis.txt", "a+")
        string_to_write = str()
        layers_plt = model.lstm.num_layers
        neurons_plt = model.lstm.hidden_size
        string_to_write = str(layers_plt) + (",") + str(neurons_plt)
        if mse == True:
            string_to_write = string_to_write + (",") + str(np.round(answer_mse, 4))
        else:
            string_to_write = string_to_write + (",") + str(-9999)
        if mae == True:
            string_to_write = string_to_write + (",") + str(np.round(answer_mae, 4))
        else:
            string_to_write = string_to_write + (",") + str(-9999)
        if rmse == True:
            string_to_write = string_to_write + (",") + str(np.round(answer_rmse, 4))
        else:
            string_to_write = string_to_write + (",") + str(-9999)
        string_to_write = string_to_write + (",") + ("False")
        string_to_write = string_to_write + ("\n")
        outF.write(string_to_write)
        outF.close()

def see_layers(model,
               visualize=True,
               save_fig=False):
    """
                                                    Description:
    --------------------------------------------------------------------------------------------------------------
    The function will allow to prepare the data for visualization that shows how the tensors transform and whether
    they show any signs of variation.
    --------------------------------------------------------------------------------------------------------------

                                                  Input Paramters:
    --------------------------------------------------------------------------------------------------------------
    > model     = model of a neural network being tested;
    > visualize = indicator whether the variation visualization should be outputed;
    > save_fig  = indicator whether the variation visualization should be saved;
    --------------------------------------------------------------------------------------------------------------

                                                       Output:
    --------------------------------------------------------------------------------------------------------------
    None
    --------------------------------------------------------------------------------------------------------------
    """

    # MAKE NEEDED IMPORTS
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    output = model.retrieve_signal()  # Call a method from class to get all signals
    df = pd.DataFrame(output).transpose()

    if visualize == True:
        f = plt.figure(figsize=(8, 6), dpi=80)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.title("LSTM - Variation of Unit's Output Tensors", fontsize=11)
        plt.xlabel("Prediction Instance", fontsize=11)
        plt.ylabel("Units", fontsize=11)
        plt.imshow(df, cmap=plt.get_cmap('Greens'))
        plt.show()

    if save_fig == True:
        f = plt.figure(figsize=(8, 6), dpi=80)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.title("LSTM - Variation of Unit's Output Tensors", fontsize=11)
        plt.xlabel("Prediction Instance", fontsize=11)
        plt.ylabel("Units", fontsize=11)
        plt.imshow(df, cmap=plt.get_cmap('Greens'))
        layers_plt = model.lstm.num_layers  # How many layers were used here
        neurons_plt = model.lstm.hidden_size  # How many neurons were used here
        fname = str("Variations in LSTM _ model uses {} layer(s), {} neuron(s).pdf".format(layers_plt, neurons_plt))
        f.savefig(fname)
        plt.close(f)

def nn_analysis(model,
                visualize=False,
                save_fig=False,
                prep_log=False,
                log_save=False,
                kpi=True):
    """
    Description:
    -----------------------------------------------------------------------------------------------------------
    The function will perform an analysis of the network based on the variations in the patterns of outputs that
    happen in the LSTM layer. The full analysis is done with the method of k-means analysis approach. Neurons that
    show insignificant (small to none) variation become suggested for pruning.
    -----------------------------------------------------------------------------------------------------------

    Input Parameters:
    -----------------------------------------------------------------------------------------------------------
    > model     = model from which the information will be retrieved for evaluation;
    > visualize = indicate whether the clustering plot should be visualized or not, DEFAULT: False;
    > save_fig  = indicate whether the clustering plot should be saved or not. DEFAULT: False;
    > prep_log  = indicate whether you want a full log of which neurons to prune and which ones are in which cluster
                 with their coordinates. Recommended for general overview, DEFAULT: False;
    > log_save  = will save a short log of how many neurons in which classes there are. Default: False;
    > kpi       = indicate whether an output value should be present that says how many neurons to retain for re-training.
                  DEFAULT: True;
    -----------------------------------------------------------------------------------------------------------

    Output:
    -----------------------------------------------------------------------------------------------------------
    > retain = indicates how many neurons should remain in the network if re-training happens under same conditions
    -----------------------------------------------------------------------------------------------------------
    """

    # Import packages
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    output = model.retrieve_signal()  # Call a method from class to get all signals
    df = pd.DataFrame(output).transpose()
    data_collector = np.array(df)

    # We want to find those neurons that actually deviate
    # We could try searching for them with Standard Deviations; or Find Min, Max in the sequence, and the Mean
    # Calculating the difference between mean -> max & min, will give us a 2x matrix, which we can use to calculate
    # the clusters. It's not too big of a dimensional space, but more than of a single value. Though both might be
    # experimented with.
    analysis_data = []
    for row in data_collector:
        mean_temp = row.mean()
        max_temp = row.max()
        min_temp = row.min()

        # Convert indicators above into indicators
        idx_1 = mean_temp - min_temp
        idx_2 = max_temp - mean_temp
        temp_holder = [idx_1, idx_2]

        analysis_data.append(temp_holder)

    analysis_data = np.array(analysis_data)

    # import packages for clustering and for euclidean distance calculation
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import euclidean_distances

    # We will enforce one centroid to be 0.0. Benefit = the class will be stably the same
    # The second centroid will be - the centroid of all the data points.
    # Whatever falls not in the data points centroid but of the 0.0, 0.0 location will be suggested for pruning

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=50).fit(analysis_data)  # Clustering method will be used
    kmeans_dummy = KMeans(n_clusters=1, random_state=0, n_init=50).fit(analysis_data)  # Create a all data cluster
    new_centroids = np.append(kmeans_dummy.cluster_centers_[0].reshape(-1, 2), [[0.0, 0.0]], axis=0)  # enforce needed values
    kmeans.cluster_centers_ = new_centroids

    y_kmeans = kmeans.predict(analysis_data)
    x, y = zip(*new_centroids)

    # Prepare information for the plot
    f = plt.figure(figsize=(8, 6), dpi=80)

    # Will show the centroids distance
    plt.plot(x, y, "k:", label="distance")
    plt.plot([], [], ' ', label="{}".format(np.round(euclidean_distances(new_centroids)[0][1], 4)))
    plt.scatter(new_centroids[:, 0], new_centroids[:, 1], c='green', s=100, alpha=0.5, label="centroids")

    # Will show data points
    x, y = zip(*analysis_data)
    plt.scatter(x, y, c=y_kmeans, s=50)

    plt.legend(loc="best")
    plt.title("K-means clustering of unit's output variations in 2 dimensional space", fontsize=11)
    plt.xlabel("Criteria of distance between mean and min (for std)", fontsize=11)
    plt.ylabel("Criteria of distance between mean and max (for std)", fontsize=11)
    if visualize == True:
        plt.show()
    elif visualize == False:
        plt.close()

    if save_fig == True:
        layers_plt = model.lstm.num_layers
        neurons_plt = model.lstm.hidden_size
        fname = str(
            "K-means clustering of units using {} layer(s) and {} unit(s).pdf".format(layers_plt, neurons_plt))
        f.savefig(fname)

    # Make a calculations for the log of the observations and the output
    # First we check how many and which observations are of which cluster
    log = str()
    log_calculator_1 = len(y_kmeans)
    for i, value in enumerate(y_kmeans):
        if value == 0:
            log_calculator_1 -= 1
        i += 1
        log += str(
            "   {} \t\t  {} \t\t[X: {}, Y: {}].\n".format(i, value, np.round(x[i - 1], 3), np.round(y[i - 1], 3)))
    log_calculator_0 = len(y_kmeans) - log_calculator_1

    if prep_log == True:
        print("----------------------LOG--------------------------\n")
        print("> {} neurons in class 0 \n> {} neurons in class 1.".format(log_calculator_0, log_calculator_1))
        print()
        print("Nueron#\t\tClass\t\t    Coordinates")
        print(log)
        print("---------------------------------------------------")

    if log_save == True:
        outF = open("Full Data Log _ Thesis.txt", "a+")
        outF.write("\n __Clusters Log__\n")
        outF.write("> {} neurons in class 0 \n> {} neurons in class 1.".format(log_calculator_0, log_calculator_1))
        outF.close()

    retain = log_calculator_0
    if kpi == True:
        return retain
    
    
def errors_view(neur_max,
                save_fig = False,
                file = "errors_analysis.txt"):
    """
                                            Description:
    ------------------------------------------------------------------------------------------------
    Visualize the series of errors that happen with different stages of changing the amount of neurons
    that the network works with.
    ------------------------------------------------------------------------------------------------

                                              Input:
    ------------------------------------------------------------------------------------------------
    > save_fig = indicate whether you want to save the figure produced. DEFAULT: False
    > file     = indicate the name of the file from which to read the data. DEFAULT: errors_analysis.txt
    > neur_max = amount of neurons experimented with.
    ------------------------------------------------------------------------------------------------

                                              Output:
    ------------------------------------------------------------------------------------------------
    > None = produces a plot for a visual assessment only
    ------------------------------------------------------------------------------------------------

    """

    # Import needed packages
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(file,
                     sep = ",")
    df_graph = df[df["baseline"] == False] 
    df_baseline = df[df["baseline"] == True]
    
    y_mse  = list(dict.fromkeys(df_baseline["MSE"]))
    y_mae  = list(dict.fromkeys(df_baseline["MAE"]))
    y_rmse = list(dict.fromkeys(df_baseline["RMSE"]))

    f = plt.figure(figsize=(8, 6), dpi=80)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # Evaluate data points
    plt.plot(df_graph["Neurons"], df_graph["MSE"], "bo-", label = "MSE", linewidth = 2, alpha = 0.5)
    plt.plot(df_graph["Neurons"], df_graph["MAE"], "ro-", label = "MAE", linewidth = 2, alpha = 0.5)
    plt.plot(df_graph["Neurons"], df_graph["RMSE"], "go-", label = "RMSE", linewidth = 2, alpha = 0.5)
    
    # Baseline...lines
    plt.hlines(y = y_mse, xmin = 0, xmax = neur_max, colors='b', linestyles='solid', label='Baseline MSE')
    plt.hlines(y = y_mae, xmin = 0, xmax = neur_max, colors='r', linestyles='solid', label='Baseline MAE')
    plt.hlines(y =y_rmse, xmin = 0, xmax = neur_max, colors='g', linestyles='solid', label='Baseline RMSE')
    
    plt.legend(loc = "best")
    plt.title("Errors Trend", fontsize=11)
    plt.xlabel("Quantity of Units", fontsize=11)
    plt.ylabel("Error Value", fontsize=11)
    plt.show()

    if save_fig == True:
        fname = str("Errors over Units variations.pdf")
        f.savefig(fname)