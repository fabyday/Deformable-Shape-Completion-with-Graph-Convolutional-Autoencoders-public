import tensorflow as tf 

#tf.device
def set_device(name):
    def deco(function):
        def innerdeco(*args, **kwargs):
            with tf.device(name) : 
                result = function(*args, **kwargs)
            return result
        return innerdeco
    return deco

#time check. function.
def timeset(function):
    import time 
    def deci(*args, **kwargs):
        start = time.time()
        res = function(*args, **kwargs)
        end= time.time()
        print(end-start, (function))
        return res
    
    return deci



def print_process(epoch, total_epoch, cur_batch, total_data_size, loss, save_path = "None"):
    epoch_str = "epoch : {:>3}/{:>3}".format(epoch, total_epoch)
    batch_str = "batch : {:>5}/{:>5}".format(cur_batch, total_data_size)
    save_path_str = " Saved checkpoint : {:>20}".format("Not saved yet." if save_path == None else save_path)
    loss_str = " {:>10.3e}".format(loss)

    bar_size = 20
    progressed_pos = int(cur_batch/total_data_size*bar_size)
    void_pos = (bar_size-1) - progressed_pos
    progressbar = "[{}{}{}]".format("="*progressed_pos, ">", " "*void_pos)


    tf.print(epoch_str + batch_str + loss_str + progressbar + save_path_str, end='\r')


def print_summary_detail(model):
    lines= 10*"="+"\n"
    small_lines = 10*"-"+"\n"
    model_name = lines + "model name : {}\n" 
    layer_name = small_lines + "layer name : {}\n" + small_lines
    weight_print = "weight_name : {} \n{}\n" + lines
    exec_list = model.exec_list
    
    print(model_name.format(model.name))
    for op in exec_list:
        print(layer_name.format(op.name))
        weight = op.get_weights()
        w_name = op.name
        for w in weight:
            print(weight_print.format(w_name, w))
        print(small_lines)

    
