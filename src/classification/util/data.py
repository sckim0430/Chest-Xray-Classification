"""Data Processing Utils
"""
def to_categori(array,num_class=None):
    """Convert Labels to One Hot Format Labels

    Args:
        array (np.ndarray): label array
        num_class (int, optional): num of class. Defaults to None.

    Returns:
        _type_: _description_
    """
    if num_class == None:
        print('num_class Error! You shoud give val!')
        return None

    categorical = []
    
    for a in array:
        cat = np.zeros(num_class)
        for idx in a:
            cat[idx] = 1
        categorical.append(cat)

    return categorical

def convert_to_label(Target):
    """Convert str Label to int Label

    Args:
        Target (str): label of string

    Returns:
        list: label of int list
    """
    tmp_target = []

    for t in Target:
        tmp_target.append(t.split('|'))
    
    label = []

    for t in tmp_target:
        sub_label = []
        for target in t:
            if target == 'Consolidation':
                sub_label.append(0)
            elif target == 'Pneumothorax':
                sub_label.append(1)
            elif target == 'Edema':
                sub_label.append(2)
            elif target == 'Effusion':
                sub_label.append(3)
            elif target == 'Pneumonia':
                sub_label.append(4)
            elif target == 'Cardiomegaly':
                sub_label.append(5)
            else:
                print('Target Error! you should look csv file!')
                return None
        
        sub_label.sort()

        label_sub_str = ''

        for index,val in enumerate(sub_label):
            if index == len(sub_label)-1:
                label_sub_str += '{}'.format(val)
            else:
                label_sub_str += '{}_'.format(val)

        label.append(label_sub_str)        

    return label

def preprocess_input(x, model):
    """Preprocess Input Image

    Args:
        x (np.ndarray): image array
        model (str): model name

    Returns:
        np.ndarray: image array
    """
    x = x.astype("float32")
    if model in ("inception","xception","mobilenet"): 
        x /= 255.
        x -= 0.5
        x *= 2.
    if model in ("densenet"): 
        x /= 255.
        if x.shape[-1] == 3:
            x[..., 0] -= 0.485
            x[..., 1] -= 0.456
            x[..., 2] -= 0.406 
            x[..., 0] /= 0.229 
            x[..., 1] /= 0.224
            x[..., 2] /= 0.225 
        elif x.shape[-1] == 1: 
            x[..., 0] -= 0.449
            x[..., 0] /= 0.226
    elif model in ("resnet","vgg"):
        if x.shape[-1] == 3:
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.680
        elif x.shape[-1] == 1: 
            x[..., 0] -= 115.799
    return x

def CalculateClassWeight(train_df,class_num,to_categori):
    
    labels = []
    class_frequency = []
    class_weight_dict = {}

    for lst in train_df.Labels:
        labels.append(list(int(i) for i in lst.split('_')))

    label_of_onehot = np.asarray(to_categori(labels,class_num))
    label_of_onehot = label_of_onehot.sum(axis=0)

    for each_class in range(class_num):
        class_frequency.append(label_of_onehot[each_class]/float(len(train_df)))
    for each_class in range(class_num):
        class_weight_dict[each_class] = np.max(class_frequency)/class_frequency[each_class]

    return class_weight_dict