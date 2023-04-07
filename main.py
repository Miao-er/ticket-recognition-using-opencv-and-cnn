from preprocess import *
from model import *

#设置超参数
learning_rate_letter = 0.02
momentum_letter = 0.5

learning_rate_number = 0.02
momentum_number = 0.9

epoches = 100
random_seed = 1
torch.manual_seed(random_seed)

#设置训练/测试模式  
test_mode = True #是否提供测试集，为否则使用训练集分割出验证集，仅当model_trained为False时有效
model_trained = True #是否已经有训练好的模型

#设置数据路径
train_dir = "./training_data/"
segments_dir = "./segments/"
test_dir = "./test_data/"
model_dir = "./models/"

#利用训练集构建网络
if model_trained == False:
    letter_list,letter_labels,number_list,number_labels = load_train_set(train_dir)
    LETTER_net = get_LETTER_net(letter_list,letter_labels,learning_rate_letter,momentum_letter,epoches,test_mode)
    NUM_net = get_NUM_net(number_list,number_labels,learning_rate_number,momentum_number,epoches,test_mode)
    torch.save(LETTER_net.state_dict(), model_dir + "letter_model.ckpt")
    torch.save(NUM_net.state_dict(), model_dir + "number_model.ckpt")

else:
    LETTER_net = Net(26)
    NUM_net = Net(10)
    LETTER_net.load_state_dict(torch.load(model_dir + "letter_model.ckpt"))
    NUM_net.load_state_dict(torch.load(model_dir + "number_model.ckpt"))
    
#在生成相应训练网络后，只需要执行该行语句来预测测试集
test_set(test_dir,segments_dir,LETTER_net,NUM_net)