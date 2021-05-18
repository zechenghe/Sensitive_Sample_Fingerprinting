import numpy as np


def classify(fname):
    averageImage = [129.1863, 104.7624, 93.5940]
    pix = scipy.misc.imread(fname)
    #data = np.float32(np.rollaxis(pix, 2)[::-1])

    data = np.float32(pix.copy())
    data[:,:,0] -= averageImage[2]
    data[:,:,1] -= averageImage[1]
    data[:,:,2] -= averageImage[0]
    data_out = np.array([data])
    print data_out.shape
    return data_out

def add_avg(data):
    averageImage = [129.1863, 104.7624, 93.5940]
    data[:,:,0] += averageImage[2]
    data[:,:,1] += averageImage[1]
    data[:,:,2] += averageImage[0]
    return data

def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:

    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
#    with tf.variable_scope("repeat"):
#    expanded_tensor = tf.expand_dims(tensor, -1)
#    multiples = [1] + repeats
    tiled_tensor = tf.tile(tensor, multiples = repeats)
#    repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)

#    print "Tensor shape:", tensor.get_shape().as_list()
#    print "Expanded_tensor shape", expanded_tensor.get_shape().as_list()
#    print "Multiples:", multiples
#    print "Tiled_tensor shape", tiled_tensor.get_shape().as_list()
#    print "Repeated shape: ", repeated_tesnor.get_shape().as_list()

    return tiled_tensor


def add_avg(data):

    averageImage = [129.1863, 104.7624, 93.5940]
    data[:,:,0] += averageImage[2]
    data[:,:,1] += averageImage[1]
    data[:,:,2] += averageImage[0]

    return data

def feasible_projection(data):

    averageImage = [129.1863, 104.7624, 93.5940]
    data = add_avg(data[0,:,:,:])
    data[data > 255.0] = 255.0
    data[data < 0.0] = 0.0

    data[:,:,0] -= averageImage[2]
    data[:,:,1] -= averageImage[1]
    data[:,:,2] -= averageImage[0]

    data = np.expand_dims(data, axis=0)
    return data


def diffable_softmax(logits):

    output = []
    max_x = tf.reduce_max(logits, reduction_indices=1, keep_dims=True)

    logits_shift = logits - max_x
    T = tf.reduce_sum(tf.exp(logits_shift), reduction_indices=1, keep_dims=True)

    for i in range(10):
        output.append(tf.exp(logits_shift[:,i]) / T)
    return output

def softmax_fast(x):
    max_x = np.max(x)
    x_shift = x - max_x
    T = np.sum(np.exp(x_shift))
    return np.exp(x_shift)/T


def get_var_by_name(target_W_Name):

    try:
        target_W = [v for v in tf.trainable_variables() if v.name == target_W_Name][0]
    except IndexError:
        print "No weights named " + target_W_Name

    return target_W


def get_top_k(p, k):
    p = p.reshape((-1))
    idx = p.argsort()
    idx = idx[-k:]
    return idx


def eval_gen_image(prob_eval, prob_trojaned_eval, top_k):

    vgg_top_1 = np.argmax(prob_eval)
    trojaned_top_1 = np.argmax(prob_trojaned_eval)

    print "SpeechNet recognizes it as user ", vgg_top_1
    print "Trojaned model recognizes it as ",  trojaned_top_1

    vgg_top_k = get_top_k(prob_eval, top_k)
    trojaned_top_k = get_top_k(prob_trojaned_eval, top_k)

    print "Top k SpeechNet: ", vgg_top_k, " Top k Trojaned: ", trojaned_top_k

    etp = entropy(prob_eval.T, prob_trojaned_eval.T)
    print "Top 1 Mismatch ", vgg_top_1 != trojaned_top_1, " Top k Mismatch: ", vgg_top_k != trojaned_top_k
    print "Entropy ", etp
    print "Prob origin model: ", prob_eval

    top_1_detected = (vgg_top_1 != trojaned_top_1)
    top_k_detected = (vgg_top_k != trojaned_top_k)
    return etp, top_1_detected, top_k_detected


def eval_diff(x_eval, img_init_save):
    MSE = np.sqrt(np.mean((x_eval - img_init_save)**2))
    S = np.sqrt(np.mean(img_init_save**2))
    SNR = 20*np.log10(S/MSE)
    print "SNR: ", SNR
    return SNR

def main():

#    fmodel = '/home/leo/vgg_face_caffe/VGG_FACE_deploy.prototxt'
#    fweights = './trojaned_face_model.caffemodel'
#    caffe.set_mode_cpu()
#    net = caffe.Net(fmodel, fweights, caffe.TEST)

#    net.blobs['data'].data[...] = data1
#    net.forward() # equivalent to net.forward_all()
#    prob = net.blobs['prob'].data[0].copy()
#    predict = np.argmax(prob)
#    print('classified: {0} {1}'.format(predict, prob[predict]))


    INIT_SELECTION = 'img' # 'zeros', 'uniform', 'xavier'
    NEpochs = 1000
    learningRate = 1e0
    eps = 1e-8
    top_k = 5
    save_itr = 20
    loss_th = 0.2
    SNR_th = 20     # 20dB

    save_img_dir = "./generated_img/"
    save_img_final_dir = save_img_dir + "final/"
    # Origin Set
    dir_name = "./origin_dataset/"


    #Trojaned set
    #name = "./vgg_trojaned/dataset1/Abraham_Benrubi_7_140.29_112.43_264.65_236.79.jpg"

    total_True = 0
    total_False = 0
    total_inputs = 0
    with tf.variable_scope("speech_origin") as scope_vgg:
        X = tf.Variable(tf.zeros([1, 224, 224, 3], tf.float32), name='X')

        net = VGG_Face_Net({'input': X})
        logits = net.layers['fc8']

        prob = tf.nn.softmax(logits)
        z = diffable_softmax(logits)
        f = 0
        for i in range(10):
            f = f + tf.log(z[i] + eps)

        target_W = get_var_by_name("speech_origin/fc8/weights:0")

        df_dw = tf.gradients(f, [target_W])
        loss = tf.nn.l2_loss(df_dw) / tf.cast(2*tf.size(target_W), tf.float32)

        optimizer = tf.train.AdamOptimizer(learningRate).minimize(-loss, var_list=[X])


    # Set up trojaned network
    with tf.variable_scope("trojaned") as scope_trojaned:
        X_trojaned = tf.Variable(tf.zeros([1, 224, 224, 3], tf.float32), name='X_trojaned')
        net_trojaned = VGG_Face_Net({'input': X_trojaned})
        logits_trojaned = net_trojaned.layers['fc8']
        prob_trojaned = tf.nn.softmax(logits_trojaned)


    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # Load weights
        with tf.variable_scope("speech_origin") as scope_vgg:
            net.load('./vgg_face/vgg_face.npy', sess)

        with tf.variable_scope("trojaned") as scope_trojaned:
            net_trojaned.load('./vgg_trojaned_wm/trojaned_wm.npy', sess)


        for filename in os.listdir(dir_name)[100:]:
            os.system("nvidia-smi")

            # Load data
            entropy_hist = []
            SNR_hist = []

            # Select Initialization
            print dir_name + filename
            if (INIT_SELECTION == 'img'):
                img = classify(dir_name + filename)
            elif (INIT_SELECTION == 'uniform'):
                img = np.random.uniform(low=0.0, high=255.0, size=[1, 224, 224, 3])
            elif (INIT_SELECTION == 'xavier'):
                img = np.random.normal(low=0.0, scale=20.0, size=[1, 224, 224, 3])
            elif (INIT_SELECTION == 'zeros'):
                img = np.zeros([1, 224, 224, 3], dtype=np.float32)
            else:
                print "Invalid Init"
                exit()

            img_init_save = img


            X.load(img)
            X_trojaned.load(img)

            # Feed init images, should be the same for non-trigger images
            prob_eval = prob.eval()
            prob_trojaned_eval = prob_trojaned.eval()

            etp, top_1_detected, top_k_detected = eval_gen_image(prob_eval, prob_trojaned_eval, top_k)
            x_eval = X.eval()

            last_loss = 0
            for i in range(NEpochs+1):

                print "Iteration ", i, " File " + filename
                x_eval_last_time = x_eval

                _, l = sess.run([optimizer, loss])
                x_eval = X.eval()

                # Projection
                x_eval = feasible_projection(x_eval)

                # Feed gen image
                X.load(x_eval)
                X_trojaned.load(x_eval)

                prob_eval = prob.eval()
                prob_trojaned_eval = prob_trojaned.eval()

                etp, top_1_detected, top_k_detected = eval_gen_image(prob_eval, prob_trojaned_eval, top_k)
                loss_eval = loss.eval()
                print "Loss ", loss_eval
                x_eval = X.eval()
                entropy_hist.append(etp)

                SNR = eval_diff(x_eval, img_init_save)
                SNR_hist.append(SNR)

                if (i == 0) and (loss_eval == 0):
                    break

                # If gen image deviate from origin, stop and save
                if (loss_eval > loss_th):
                    if eval_diff(x_eval, img_init_save) > SNR_th:
                        if not os.path.exists(save_img_final_dir):
                            os.makedirs(save_img_final_dir)

                        # Feed last gen image
                        X.load(x_eval_last_time)
                        X_trojaned.load(x_eval_last_time)

                        prob_eval = prob.eval()
                        prob_trojaned_eval = prob_trojaned.eval()

                        etp, top_1_detected, top_k_detected = eval_gen_image(prob_eval, prob_trojaned_eval, top_k)
                        img_gen = add_avg(x_eval[0,:,:,:]) / 255.0
                        #plt.imshow(img_gen)
                        #plt.axis('off')
                        #plt.savefig( save_img_final_dir + filename + "_" + str(i) + "_" + str(top_1_detected) + '.jpg')
                        scipy.misc.imsave(save_img_final_dir + filename + "_" + str(i) + "_" + str(top_1_detected) + '.jpg', img_gen)

                        if (top_1_detected):
                            total_True += 1
                        else:
                            total_False += 1

                    total_inputs += 1
                    print "Total detected: ", total_True, "Total undetected: ", total_False, " Total inputs: ", total_inputs

                    #print "Image " + filename + " top 1 can detect trojan: ", top_1_detected
                    #print "Image " + filename + " top k can detect trojan: ", top_k_detected
                    break

                if (i % save_itr) == 0:

                    if not os.path.exists(save_img_dir):
                        os.makedirs(save_img_dir)

                    img_gen = add_avg(x_eval[0,:,:,:]) / 255.0
                    x_max = np.max(img_gen)
                    x_min = np.min(img_gen)
                    print " Max ", x_max, " Min ", x_min
                    #plt.imshow(img_gen, cmap="gray")
                    #plt.axis('off')
                    #plt.savefig( save_img_dir + filename + "_" + str(i) + '.jpg')
                    scipy.misc.imsave(save_img_dir + filename + "_" + str(i) + '.jpg', img_gen)

                    if (i == NEpochs) and (loss_eval > loss_th):
                        if not os.path.exists(save_img_final_dir):
                            os.makedirs(save_img_final_dir)

                        img_gen = add_avg(x_eval[0,:,:,:]) / 255.0
                        #plt.imshow(img_gen)
                        #plt.axis('off')
                        #plt.savefig( save_img_final_dir + filename + "_" + str(i) + "_" + str(top_1_detected) + '.jpg')
                        scipy.misc.imsave(save_img_final_dir + filename + "_" + str(i) + "_" + str(top_1_detected) + '.jpg', img_gen)
                        print "Image " + filename + " top 1 can detect trojan: ", top_1_detected
                        print "Image " + filename + " top k can detect trojan: ", top_k_detected

                last_loss = loss_eval
                # End for i in range(NEpochs+1):

            np.savetxt(save_img_dir + filename + '_entropy.csv', entropy_hist , fmt='%.3f')
            np.savetxt(save_img_dir + filename + '_SNR.csv', SNR_hist , fmt='%.3f')

if __name__ == '__main__':
    main()
