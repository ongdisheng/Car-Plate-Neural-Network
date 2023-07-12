# import libraries
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

# TRAIN
# step 1
def Weight_Initialization(num_input, num_hidden, num_output):
    # initialize weight and bias
    # random float number between -0.5 to 0.5 (weight)
    np.random.seed(31109667)
    wji= np.random.uniform(-0.5, 0.5, size=(num_hidden, num_input))
    wkj = np.random.uniform(-0.5, 0.5, size=(num_output, num_hidden))
    bias_j = np.random.uniform(0, 1, size=(num_hidden, 1))
    bias_k = np.random.uniform(0, 1, size=(num_output, 1))

    return wji, wkj, bias_j, bias_k

# step 2
def Read_Files():
    # read segmented training files and target files

    # initialize train and test data list
    train_data = []
    test_data = []

    # initialize train and test label list
    train_label = []
    test_label = []

    # iterate over each folder
    for label, dir in enumerate(os.listdir('data')):

        # iterate over each image
        for index, img_path in enumerate(glob.glob(os.path.join('data', dir, '*.jpg'))):

            # read image
            img = cv2.imread(img_path)

            # convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

            # resize image
            img_gray = cv2.resize(img_gray, (20, 40))

            # binary thresholding
            _, img_bin = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY)

            # remove small noise
            contours, _ = cv2.findContours(img_bin, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
            for c in contours:
                cv2.drawContours(img_bin, [c], -1, (0,0,0), -1)
            
            # optional
            # clip image
            img_clip = np.clip(img_bin, 0, 1)

            # reshape image
            img_final = np.reshape(img_clip, (1, img_clip.shape[0] * img_clip.shape[1]))

            # train
            if index < 8:
                train_data.append(img_final)
                train_label.append(label)
            
            # test 
            else:
                test_data.append(img_final)
                test_label.append(label)

            # # show image
            # cv2.imshow('window', img_bin)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    # shuffle train data and label (a1-z1; ...; a8-z8)
    # train_data_shuffle = []
    # train_label_shuffle = []
    # for i in range(8):

    #     ## MODIFY ##
    #     for j in range(20):
    #         train_data_shuffle.append(train_data[i + j * 8])
    #         train_label_shuffle.append(train_label[i + j * 8])
    
    # # convert label to one hot vector
    train_label = np.eye(np.max(train_label) + 1)[train_label]
    test_label = np.eye(np.max(test_label) + 1)[test_label]

    return train_data, train_label, test_data, test_label

# sigmoid activation function
def sigmoid(x, b):
    return 1 / (1 + np.exp(-(x + b)))

# step 3
def Forward_Input_Hidden(X, W1, b1):
    # hidden layer
    Z1 = np.dot(W1, np.transpose(X))      # NetJ
    A1 = sigmoid(Z1, b1)                  # OutJ
    return A1 

# step 4
def Forward_Hidden_Output(A1, W2, b2):
    # output layer
    Z2 = np.dot(W2, A1)     # NetK
    A2 = sigmoid(Z2, b2)    # OutK
    return np.transpose(A2) 

# step 5
def Check_For_End(total_error, thresh_error, iter, max_iter):
    # condition met
    if total_error < thresh_error or iter == max_iter:
        return True
    
    return False

# step 6
def Weight_Bias_Correction_Output(A1, A2, Y):
    # correction of weights and bias between hidden and output layer
    delta_Z2 = (A2 - Y) * (A2 * (1 - A2))
    delta_W2 = np.dot(np.transpose(delta_Z2), np.transpose(A1))
    delta_b2 = np.sum(np.transpose(delta_Z2), axis=1, keepdims=True)
    return delta_W2, delta_b2

# step 7
def Weight_Bias_Correction_Hidden(A1, A2, W2, X, Y):
    # correction of weights and bias between input and hidden layer
    delta1 = np.transpose(X)
    delta2 = A1 * (1 - A1)
    delta3 = (A2 - Y) * (A2 * (1 - A2))
    delta4 = W2
    temp = delta3.dot(delta4)
    temp = np.transpose(delta2) * temp
    delta_WJ = np.transpose(delta1.dot(temp))
    delta_biasJ = np.sum(np.transpose(temp), axis = 1)
    delta_biasJ = np.expand_dims(delta_biasJ, axis=1)
    return delta_WJ, delta_biasJ     


# step 8
def Weight_Bias_Update(dic):
    # update weights and bias
    W1_new = dic["W1"] - dic["alpha"] * dic["dW1"]
    b1_new = dic["b1"] - dic["alpha"] * dic["db1"]
    W2_new = dic["W2"] - dic["alpha"] * dic["dW2"]
    b2_new = dic["b2"] - dic["alpha"] * dic["db2"]
    return W1_new, W2_new, b1_new, b2_new

# step 10
def Saving_Weights_Bias(W1, W2, b1, b2):
    # create checkpoint directory
    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")

    # save weights and bias 
    np.save(os.path.join('checkpoint', 'W1'), W1)
    np.save(os.path.join('checkpoint', 'W2'), W2)
    np.save(os.path.join('checkpoint', 'b1'), b1)
    np.save(os.path.join('checkpoint', 'b2'), b2)

# main function for training neural network
def train():
    ## MODIFY ##
    INPUT_NEURONS = 800     # number of neurons (input layer) 
    HIDDEN_NEURONS = 128    # number of neurons (hidden layer)
    OUTPUT_NEURONS = 20     # number of neurons (output layer)
    NUM_OF_ITER = 5000      # number of training iteration
    MAX_ITER = 5000         # maximum number of training iteration
    ALPHA = 0.01            # learning rate
    THRESH_ERROR = 0.001    # error threshold 

    # initialize weight
    W1, W2, b1, b2 = Weight_Initialization(INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS) 

    # read files
    train_data, train_label, _, _ = Read_Files()

    # transpose data and label
    X_train = np.vstack(train_data)
    Y_train = train_label
    # X_test = np.vstack(test_data).T
    # Y_test = test_label.T

    # train phase 
    # iterate over iter number of times
    for i in range(NUM_OF_ITER):

        # forward propagation from input to hidden layer
        A1 = Forward_Input_Hidden(X_train, W1, b1)

        # forward propagation from hidden to output layer
        A2 = Forward_Hidden_Output(A1, W2, b2)
        
        # compute total error
        total_error = 0.5 * np.sum(np.square(Y_train - A2))

        # log total error to console
        print('Iter: {} ======== Error: {}'.format(i+1, total_error))

        # validate global error or number of iterations 
        if Check_For_End(total_error, THRESH_ERROR, i, MAX_ITER):
            break

        # back propagation 
        # weights and bias between hidden and output layer
        delta_W2, delta_b2 = Weight_Bias_Correction_Output(A1, A2, Y_train)

        # weights and bias between input and hidden layer
        delta_W1, delta_b1 = Weight_Bias_Correction_Hidden(A1, A2, W2, X_train, Y_train)

        # create dictionary storing information related to weight and bias update
        dic =  {
            "W1": W1,
            "W2": W2,
            "b1": b1,
            "b2": b2,
            "dW1": delta_W1,
            "dW2": delta_W2,
            "db1": delta_b1,
            "db2": delta_b2,
            "alpha": ALPHA
        }

        # update weights and bias
        W1, W2, b1, b2 = Weight_Bias_Update(dic)

    # save weights and bias
    Saving_Weights_Bias(W1, W2, b1, b2)

# TEST
# step 1
def Loading_Weights_Bias():
    W1 = np.load(os.path.join('checkpoint', 'W1.npy'))
    W2 = np.load(os.path.join('checkpoint', 'W2.npy'))
    b1 = np.load(os.path.join('checkpoint', 'b1.npy'))
    b2 = np.load(os.path.join('checkpoint', 'b2.npy'))
    return W1, W2, b1, b2

# main function for testing neural network
def test():
    
    # load weights and bias
    W1, W2, b1, b2 = Loading_Weights_Bias()

    # read files 
    _, _, test_data, test_label = Read_Files()

    # transpose data and label
    X_test = np.vstack(test_data)
    Y_test = test_label

    # forward propagation from input to hidden layer
    A1 = Forward_Input_Hidden(X_test, W1, b1)

    # forward propagation from hidden to output layer
    A2 = Forward_Hidden_Output(A1, W2, b2)

    # initialize match counter 
    match = 0
    idx2alpha = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 'B',
        11: 'F',
        12: 'L',
        13: 'M',
        14: 'P',
        15: 'Q',
        16: 'T',
        17: 'U',
        18: 'V',
        19: 'W'
    }
    number_of_images_satisfied_criteria = 0
    # test phase 
    for i in range(40):
        pred = A2[i, :]
        true = Y_test[i, :]
        pred_idx = np.argmax(pred)
        true_idx = np.argmax(true)


        # match
        if pred_idx == true_idx:
            match += 1

        # output predicted and actual character for each test image
        print('Test Image ' + str(i+1))
        print("Predict:", idx2alpha[pred_idx])
        print("Actual:", idx2alpha[true_idx])
        print('-' * 20)

        # optional code to calculate how many within the range of output value/ probability
        # if 0.9 <= pred[pred_idx] < 1:
        #     number_of_images_satisfied_criteria += 1

    # log test accuracy to console
    print("Accuracy (Manually Cropped):", match, "/", "40", "=", round(match/40*100,3), "%")
    print('-' * 20)

# SEGMENT
# step 1
def consecutive(data, step_size=1):
    """
    Retrieve groups of consecutive numbers in data
    e.g. [0, 47, 48, 49, 50, 97, 98, 99] => [(0), (47, 48, 49, 50), (97, 98, 99)]
    """
    return np.split(data, np.where(np.diff(data) != step_size)[0] + 1)

# step 2
def split(data):

    # compute length for each group 
    # each entry represents a character in the plate
    length_x = [len(x) for x in data]

    # some characters might not have been split nicely
    if len(length_x) < 7:

        max_val = max(length_x)
        max_idx = length_x.index(max_val)
        min_val = min(length_x)
        min_idx = length_x.index(min_val)

        # found
        if max_val / min_val > 2:
            left = data[max_idx][:min_val]
            right = data[max_idx][min_val:]
            data.pop(max_idx)
            data.insert(max_idx, left)
            data.insert(max_idx + 1, right)
    
    return data

# segment plate images 
def segment():

    # iterate over each plate image 
    for img_path in glob.glob(os.path.join('plate', '*.jpg')):
        
        # read image
        img = cv2.imread(img_path)

        # convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        # binary thresholding
        _, img_bin = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY)

        # remove small noise
        contours, _ = cv2.findContours(img_bin, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[7:]
        for c in contours:
            cv2.drawContours(img_bin, [c], -1, (0,0,0), -1)

        # vertical projection
        img_proj = img_bin.copy()
        img_proj[img_proj == 255] = 1
        proj_v = np.sum(img_proj, axis=0)

        # non zero (vertical projection) 
        non_zero_x = np.where(proj_v != 0)[0]
        non_zero_group_x = consecutive(non_zero_x)
        non_zero_group_x = split(non_zero_group_x)
        
        # create segmented directory
        dir_name = os.path.join("segmented", os.path.splitext(os.path.split(img_path)[1])[0])
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # initialize segment counter
        counter = 1

        # begin segmentation
        for x in non_zero_group_x:
            x_min = x[0] - 2
            x_max = x[-1] + 2
            img_segment = img[:, x_min:x_max]
            cv2.imwrite(os.path.join(dir_name, '{}.jpg'.format(counter)), img_segment)
            counter += 1

# PREDICT
# step 1
def preprocess():
    # initialize data list
    X = []
    
    # iterate over each folder
    for dir in os.listdir('segmented'):
        
        # iterate over each image
        for img_path in glob.glob(os.path.join('segmented', dir, '*.jpg')):
            
            # read image
            img = cv2.imread(img_path)

            # convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # resize image
            img_gray = cv2.resize(img_gray, (20, 40))

            # binary thresholding
            _, img_bin = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY)

            # remove small noise
            contours, _ = cv2.findContours(img_bin, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
            for c in contours:
                cv2.drawContours(img_bin, [c], -1, (0,0,0), -1)

            # horizontal projection
            img_proj = img_bin.copy()
            img_proj[img_proj == 255] = 1
            proj_h = np.sum(img_proj, axis=1)

            # non zero (horizontal projection)
            non_zero_y = np.where(proj_h != 0)[0]
            y_min = non_zero_y[0] - 1
            y_max = non_zero_y[-1] + 1
            img_bin = img_bin[y_min:y_max, :]
            img_bin = cv2.resize(img_bin, (20, 40))

            # clip image
            img_clip = np.clip(img_bin, 0, 1)
            
            # reshape image
            img_final = np.reshape(img_clip, (1, img_clip.shape[0] * img_clip.shape[1]))

            X.append(img_final)

            # cv2.imshow('window', img_bin)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    return X

# main function for predicting segmented car plate images
def predict():

    # load weights and bias
    W1, W2, b1, b2 = Loading_Weights_Bias()

    # preprocess segmented images
    X = preprocess()
    
    # transpose data
    X = np.vstack(X)
    
    # forward propagation from input to hidden layer
    A1 = Forward_Input_Hidden(X, W1, b1)

    # forward propagation from hidden to output layer
    A2 = Forward_Hidden_Output(A1, W2, b2)

    # initialize counter
    counter = 0

    # used to convert index to alphabet
    idx2alpha = {
        10: 'B',
        11: 'F',
        12: 'L',
        13: 'M',
        14: 'P',
        15: 'Q',
        16: 'T',
        17: 'U',
        18: 'V',
        19: 'W'
    }

    # Actual number plates
    actual = [["V","B","U",3,8,7,8],["V","B","T",2,5,9,7],["W","T","F",6,8,6,8],["P","L","W",7,9,6,9],["B","P","U",9,8,5,9],["B","M","T",8,6,2,8],["B","M","B",8,2,6,2],["P","P","V",7,4,2,2],["B","Q","P",8,1,8,9],["W","U","M",2,0,7]]

    # initialize accuracy variables
    total_char = 0
    total_alpha = 0
    total_num = 0
    correct_char = 0
    correct_alpha = 0
    correct_num = 0

    # iterate over each folder
    for dir in os.listdir('segmented'):

        # log current image to console 
        print('Image {}'.format(dir))

        # actual number plate
        print('Actual:', actual[int(dir)-1])

        # iterate over each segmented image
        for j, _ in enumerate(glob.glob(os.path.join('segmented', dir, '*.jpg'))):
            
            # retrieve prediction probability 
            pred = A2[counter, :]
            
            if j < 3:
                print('Predicted:', idx2alpha[np.argmax(pred[10:]) + 10])
                total_char += 1
                total_alpha += 1
                if idx2alpha[np.argmax(pred[10:]) + 10] == actual[int(dir)-1][j]:
                    correct_char += 1
                    correct_alpha += 1
            else:
                print('Predicted:', np.argmax(pred[:10]))
                total_char += 1
                total_num += 1
                if np.argmax(pred[:10]) == actual[int(dir)-1][j]:
                    correct_char += 1
                    correct_num += 1
            
            counter += 1

        print('-' * 50)

    # log test accuracy to console
    print("Accuracy (Segmented)")
    print("Total:", correct_char, "/", total_char, "=", correct_char/total_char,  "=", round(correct_char/total_char*100, 3), "%")
    print("Alphabet:", correct_alpha, "/", total_alpha, "=", correct_alpha/total_alpha,  "=", round(correct_alpha/total_alpha*100, 3), "%")
    print("Number:", correct_num, "/", total_num, "=", correct_num/total_num,  "=", round(correct_num/total_num*100, 3), "%")

if __name__ == "__main__":

    # train neural network
    train()

    # test neural network
    test()

    # segment car plate images
    segment()

    # predict segmented car plate images
    predict()