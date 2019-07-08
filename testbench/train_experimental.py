from models_experimental import * 


def train_model(model_type, X, Y, param_dict=None):
    # call different trainer functions based off of the input arguments
    # parameter_dict:
    #   epochs, steps, batch_size, lr, loss, dropout, kernel_num, kernel_sizes (cnntext)
    if model_type is not None and parameter_dict is None:
        raise #error of some kind
    if model_type == 'cnntext':
        # X -> cnntext_data
        cnn_model = mdl.CNNText(300, len(param_dict['unique_labels']), dropout=param_dict['dropout_rate'], ensemble=False)
        cnn_opt = torch.optim.Adam(cnn_model.parameters(), lr=param_dict['lr'])
        batch_size = param_dict['batch_size']
        epoch_num = param_dict['epoch_num']
        index_chunk_list = list(chunks(list(range(0, X.shape[0])), batch_size))
        return train_cnntext(cnn_model, X, Y, epoch_num, index_chunk_list, cnn_opt)
    elif model_type == 'cnnrnn':
        # X -> cnntext_data 
        cnn_model = mdl.CNNText(300, len(param_dict['unique_labels']), dropout=param_dict['dropout_rate'], ensemble=True)
        rnn_model = mdl.TextRNNClassifier(param_dict['kernel_num'] * param_dict['kernel_sizes'], hidden_size = param_dict['hidden_size'], output_size = len(param_dict['unique_labels']))
        cnn_opt = torch.optim.Adam(cnn_model.parameters(), lr=param_dict['lr'])
        rnn_opt = torch.optim.Adam(rnn_model.parameters(), lr=param_dict['lr'])
        batch_size = param_dict['batch_size']
        epoch_num = param_dict['num_epochs']
        index_chunk_list = list(chunks(list(range(0, X.shape[0])), batch_size))
        return train_cnnrnn([cnn_model, rnn_model], X, Y, epoch_num, index_chunk_list, [cnn_opt, rnn_opt])
    elif model_type == 'basernn':
        # X -> input_matrix 
        base_rnn = mdl.BaseRNN(len(list(param_dict['deva_index'].keys())), 300, param_dict['embed_mat'], len(param_dict['unique_labels']))
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, base_rnn.parameters()), lr=param_dict['lr'], weight_decay=param_dict['weight_decay'])
        batch_size = param_dict['batch_size']
        epoch_num = param_dict['num_epochs']
        index_chunk_list = list(chunks(list(range(0, X.shape[0])), batch_size))
        return train_basernn(base_rnn, X, Y, epoch_num, index_chunk_list, optimizer)
    else:
        raise # error 

def train_cnntext(model, X, Y, epoch_num, index_chunk_list, cnn_opt):
    # use_cuda = 0
    criterion = nn.CrossEntropyLoss()

    cnn_model.train()

    for i in range(epoch_num):
        print('current epoch: {}'.format(str(i)))
        for index_chunk in index_chunk_list:

            X_batch = X[index_chunk, :, :, :]
            y_batch = Y[index_chunk]

            print(X_batch.shape, y_batch.shape)

            features = torch.FloatTensor(X_batch)
            target = torch.LongTensor(y_batch)

            # print(type(features), type(target))

            cnn_opt.zero_grad()

            cnn_out = model(features)
            # print(target.shape)
            # print(cnn_out.shape)

            print(cnn_out.shape)
            # checker = [j for j in range(batch_size) if z + j < cnntext_data.shape[0]]
            # print(checker)
            # pred_labels_tuple = torch.max(basernn_out, dim=1)
            # pred_labels = pred_labels_tuple[1] 
            # print(target)
            # print(pred_labels)
            # one at a time ...
            loss = criterion(cnn_out, target)
            loss.backward()
            cnn_opt.step()

    return model

def train_cnnrnn(model_ensemble_list, X, Y, epoch_num, opt_list):
    # use_cuda = 0
    criterion = nn.CrossEntropyLoss()

    cnn_model = model_ensemble_list[0]
    rnn_model = model_ensemble_list[1]

    cnn_opt = opt_list[0]
    rnn_opt = opt_list[1]

    cnn_model.train()
    rnn_model.train()

    for i in range(epoch_num):
        print('current epoch: {}'.format(str(i)))
        for index_chunk in index_chunk_list:

            X_batch = cnntext_data[index_chunk, :, :, :]
            y_batch = labels[index_chunk]

            print(X_batch.shape, y_batch.shape)

            features = torch.FloatTensor(X_batch)
            target = torch.LongTensor(y_batch)

            # print(type(features), type(target))

            cnn_opt.zero_grad()

            cnn_out = cnn_model(features)
            # print(target.shape)
            # print(cnn_out.shape)

            print(cnn_out.shape)
            # checker = [j for j in range(batch_size) if z + j < cnntext_data.shape[0]]
            # print(checker)
            # pred_labels_tuple = torch.max(basernn_out, dim=1)
            # pred_labels = pred_labels_tuple[1] 
            # print(target)
            # print(pred_labels)
            # one at a time ...
            for j in range(cnn_out.shape[0]):
                rnn_out = rnn_model(cnn_out[j], 1)
                loss = F.cross_entropy(rnn_out, torch.argmax(target[j]).reshape((1,)))
            # loss = criterion(cnn_out, target)
            loss.backward()
            cnn_opt.step()
            rnn_opt.step()
    return cnn_model, rnn_model

def train_basernn(model, X, Y, epoch_num, optimizer):
    use_cuda = 0
    criterion = nn.CrossEntropyLoss()
    model.train()
    for i in range(epoch_num):
        print('current epoch: {}'.format(str(i)))
        for index_chunk in index_chunk_list:

            X_batch = X[index_chunk, :]
            y_batch = Y[index_chunk]

            print(X_batch.shape, y_batch.shape)

            features = torch.LongTensor(X_batch)
            target = torch.LongTensor(y_batch)

            # print(type(features), type(target))

            optimizer.zero_grad()

            basernn_out = model(features)
            print(target.shape)
            print(basernn_out.shape)
            # pred_labels_tuple = torch.max(basernn_out, dim=1)
            # pred_labels = pred_labels_tuple[1] 
            # print(target)
            # print(pred_labels)
            loss = criterion(basernn_out, target)
            loss.backward()
            optimizer.step()
    return model

def save_model(model, location):
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    print('saving model state_dict to: {}'.format(str(location)))
    torch.save(model.state_dict(), location)
    print('saved.')
    return 
    
'''
def train_model(model_type, X, Y, parameter_dict=None):
    # call different trainer functions based off of the input arguments
    # parameter_dict:
    #   epochs, steps, batch_size, lr, loss, dropout, kernel_num, kernel_sizes (cnntext)
    if model_type is not None and parameter_dict is None:
        raise #error of some kind
    if model_type == 'cnntext':
        return 
    elif model_type == 'encdecrnn'
        return
    else:
        raise # error 
'''

'''
def train_cnntext(X, Y, parameter_dict):

    # data to arrays
    X_array = np.asarray(X).astype('float') # input WORD EMBEDDING DATA
    Y_array = Y.astype('int') # classes

    # parameters
    X_len = X_array.shape[0]
    dim = X_array.shape[-1]
    n_labels = Y_array.shape[-1]
    n_epochs = parameter_dict[epochs]
    steps = parameter_dict[steps]
    batch_size = parameter_dict[batch_size]
    num_batches = math.ceil(X_len / batch_size)
    learning_rate = parameter_dict[lr]

    n_hidden = parameter_dict[hidden]

    kernel_num = parameter_dict[kern_num]
    kernel_sizes = parameter_dict[kern_sizes]
    dropout = parameter_dict[dropout]


    # instantiate models
    CNN = CNNText(dim, n_labels, kernel_num=kernel_num, dropout=dropout, kernel_sizes=kernel_sizes, ensemble=True)
    RNN = RNNClassifier(input_size = kernel_num * kernel_sizes, hidden_size = n_hidden, output_size=n_labels)

    if parameter_dict[cuda]:
        CNN = CNN.cuda()
        RNN = RNN.cuda()

    # set start time 
    st = time.time()

    # define optimizers
    CNN_opt = torch.optim.Adam(CNN.parameters(), lr = learning_rate)
    RNN_opt = torch.optim.Adam(RNN.parameters(), lr = learning_rate)

    # training-mode cnn 
    CNN.train()

    # randomize data 
    np.random.seed(seed=1)
    permute = torch.from_numpy(np.random.permutation(X_len)).long()
    X_iter = X_array[permutation]
    Y_iter = Y_array[permutation]

    # training cycle

    i = 0

    while i + batch_size < X_len :

        # batch of data
        batch_X = X_iter[i : i + batch_size]
        batch_Y = Y_iter[i : i + batch_size]

        # tensorize and set as feat/target vars
        X_tensor = torch.from_numpy(batch_X).float()
        Y_tensor = torch.from_numpy(batch_Y).long()
        feature = Variable(X_tensor)
        target = Variable(Y_tensor)

        # define optimizers at zero grad
        CNN_opt.zero_grad()
        RNN_opt.zero_grad()

        # train CNN
        CNN_out = CNN(feature)
        print('CNN trained')
        print('Output size: {}'.format(str(CNN_out.size())))

        # train RNN one-by-one
        for j in range(batch_size):
            if i + j < X_len:
                RNN_out = RNN(CNN_out[j], 1)
                loss = F.cross_entropy(RNN_out, torch.argmax(target[j]).reshape((1,)))
        print('RNN trained')

        loss.backward()
        CNN_opt.step()
        RNN_opt.step()

        steps += 1
        i = i + batch_size 

        # done while loop

    # print epoch time:
    ct = time.time() - st 
    print('time thus far: {} s'.format(str(ct)))

    return CNN, RNN
    
def train_basicrnn(X, Y, parameter_dict):
    return

def train_attnrnn(X, Y, parameter_dict):
    return
'''
'''
def train_textcnnrnn(X, Y, parameter_dict):

    X_a = np.asarray(X).astype('float')
    Y_a = Y.astype('int')

    # parameter dict:
    #   num_epochs
    #   lr
    #   kernel_num
    #   kernel_sizes
    #   dropout_rate
    #   use_cuda
    #   hidden_size

    dim = X_a.shape[-1]
    num_labels = Y_a.shape[-1]

    steps = 0
    batch_size = parameter_dict['batch_size']
    num_batches = match.ceil(X_a.shape[0] / batch_size)
    num_epochs = parameter_dict['num_epochs']
    learning_rate = parameter_dict['lr']

    kernel_num = parameter_dict['kernel_num']
    kernel_sizes = parameter_dict['kernel_sizes']
    dropout_rate = parameter_dict['dropout_rate']
    hidden_num = parameter_dict['hidden_size']

    # model instantiation

    cnn = TextCNN(
        dim,
        num_labels,
        kernel_num=kernel_num,
        dropout=dropout_rate,
        kernel_sizes=kernel_sizes,
        ensemble=True
    )
    rnn = TextRNNClassifier(
        kernel_num * kernel_sizes,
        hidden_size = hidden_num,
        output_size = parameter_dict['num_labels']
    )

    if parameter_dict['use_cuda']:
        cnn = cnn.cuda()
        rnn = rnn.cuda()

    # optimizers

    cnn_opt = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    rnn_opt = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    cnn.train()
    np.random.seed(seed=1)
    permute = torch.from_numpy(np.random.permutation(X_a.shape[0])).long()

    X_iter = X_a[permute]
    Y_iter = Y_a[permute]

    i = 0
    while i + batch_size < X_a.shape[0]:

        batch_X = X_iter[i:i+batch_size]
        batch_Y = Y_iter[i:i+batch_size]
        X_t = torch.from_numpy(batch_X).float()
        Y_t = torch.from_numpy(batch_Y).long()

        feature = X_t
        target = Y_t

        cnn_opt.zero_grad()
        rnn_opt.zero_grad()

        # training TextCNN
        cnn_output = cnn(feature)

        # train TextRNNClassifier one by one
        for j in range(batch_size):
            if i + j < X_a.shape[0]:
                rnn_output = rnn(cnn_output[j], 1)
                loss = F.cross_entropy(rnn_output, torch.argmax(target[j]).reshape((1,)))

        loss.backward()
        cnn_opt.step()
        rnn_opt.step()
        steps += 1
        i = i + batch_size 

    return cnn, rnn  

def train_basernn(X, Y, loss, parameter_dict):

    rnn = BaseRNN(
        vocab_size = parameter_dict['vocab_size'],
        embed_size = parameter_dict['embed_size'],
        embed_matrix = parameter_dict['embed_matrix'],
        output_num = parameter_dict['num_outputs'],
        hidden_size = parameter_dict['num_hiddens'],
        pad_idx = 0,
        num_layers = parameter_dict['num_layers']
    )

    criterion = loss # e.g. nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameter_dict['lr'], weight_decay=parameter_dict['weight_decay'])

    if parameter_dict['use_cuda']:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True
        model.cuda()
        criterion = criterion.cuda()
    
    num_epochs = parameter_dict['num_epochs']

    for epochs in range(num_epochs):
        for i, data in enumerate(X, Y): # need to fix this
            batch_X = data['X']
            batch_Y = data['Y']

            # convert X_t and Y_t from batch_X and batch_Y using the above

            features = Variable(X_t)
            target = Variable(Y_t)

            optimizer.zero_grad()

            basernn_out = rnn(features)
            loss = criterion(basernn_out, target)
            loss.backward()
            optimizer.step()

    return rnn 

def save_model(model, location):
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    print('saving model state_dict to: {}'.format(str(location)))
    torch.save(model.state_dict(), location)
    print('saved.')
    return 

'''