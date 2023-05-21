
def generate_pair(path):
    
    #path = '/home/ubuntu/Desktop/data/pjh/images/images/'
    #config = configs.models_genesis_config
    #path = r'/home/ubuntu/Desktop/data/pjh/images/images'
    #files = os.listdir(targetdir)
    #condition='*.png'
    #fileExt = r".png"
    #onlyfiles = [os.path.join(targetdir,_) for _ in os.listdir(targetdir) if _.endswith(fileExt)]
    #onlyfiles = onlyfiles[0:10]
    #for i in range(len(onlyfiles)):
    config = configs.models_genesis_config

    img = Image.open(path)

    img = img.convert('RGB')

    img = img.resize((512,512))

    #img= tf.keras.preprocessing.image.img_to_array(img)

    img = np.array(img)

    img_rows, img_cols, img_deps = img.shape[0], img.shape[1], img.shape[2]

    while True:

        y = img/255

        x = copy.deepcopy(y)            

        # Autoencoder

        x = copy.deepcopy(y)

            

        # Flip

        x, y = data_augmentation(x, y, config.flip_rate)

 

        # Local Shuffle Pixel

        x = local_pixel_shuffling(x, prob=config.local_rate)

            

        # Apply non-Linear transformation with an assigned probability

        x = nonlinear_transformation(x, config.nonlinear_rate)

            

        # Inpainting & Outpainting

        if random.random() < config.paint_rate:

            if random.random() < config.inpaint_rate:

                # Inpainting

                x = image_in_painting(x)

            else:

                # Outpainting

                x = image_out_painting(x)


        '''
        # Save sample images module

        if config.save_samples is not None and status == "train" and random.random() < 0.01:

            n_sample = random.choice( [i for i in range(conf.batch_size)] )

            sample_1 = np.concatenate((x[n_sample,0,:,2*img_deps//6].reshape(x[n].shape[0],1), y[n_sample,0,:,2*img_deps//6].reshape(x[n].shape[0],1)), axis=1)

            sample_2 = np.concatenate((x[n_sample,0,:,3*img_deps//6].reshape(x[n].shape[0],1), y[n_sample,0,:,3*img_deps//6].reshape(x[n].shape[0],1)), axis=1)

            sample_3 = np.concatenate((x[n_sample,0,:,4*img_deps//6].reshape(x[n].shape[0],1), y[n_sample,0,:,4*img_deps//6].reshape(x[n].shape[0],1)), axis=1)

            sample_4 = np.concatenate((x[n_sample,0,:,5*img_deps//6].reshape(x[n].shape[0],1), y[n_sample,0,:,5*img_deps//6].reshape(x[n].shape[0],1)), axis=1)

            final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)

            #final_sample = final_sample * 255.0

            final_sample = final_sample.astype(np.float32)

            #file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples

            #imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)

         '''

        return x, y
    

def get_nih1():
    targetdir= '/home/nextgen/Desktop/data/pjh/images/images/images'
    fileExt = r'.png'
    #onlyfiles = [os.path.join(mypath, _) for _ in os.listdir(mypath) if _.endswith(fileExt)]
    onlyfiles = [os.path.join(targetdir,_) for _ in os.listdir(targetdir) if _.endswith(fileExt)]
    onlyfiles = onlyfiles[0:-1000]
    #onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    config = configs.models_genesis_config
    
    for i in range(len(onlyfiles)):

        img1,img2 = generate_pair(onlyfiles[i])
    
        yield (img1,img2)

def get_nih2():
    targetdir= '/home/nextgen/Desktop/data/pjh/images/images/images'
    files = os.listdir(targetdir)
    condition='*.png'
    fileExt = r".png"
    #onlyfiles = [os.path.join(mypath, _) for _ in os.listdir(mypath) if _.endswith(fileExt)]
    onlyfiles = [os.path.join(targetdir,_) for _ in os.listdir(targetdir) if _.endswith(fileExt)]
    onlyfiles = onlyfiles[-1001:-1]
    #onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    config = configs.models_genesis_config
    
    for i in range(len(onlyfiles)):

        img1,img2 = generate_pair(onlyfiles[i])
    
        yield (img1,img2)