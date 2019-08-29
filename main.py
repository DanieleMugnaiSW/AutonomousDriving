import cv2 as cv
import os, csv, json
import numpy as np
import matplotlib.pyplot as plt

SCALE_PERCENT = 0.125  # Percent of original size


def adjustImages(input_path='Sequenze-Depth-RGB/', output_path='Dataset/IMG/'):
    for i in range(1, 18):  # Abbiamo 17 {1,2,...,17} sequenze di video

        if (i <= 12):    # train
            newpath = output_path + 'train/seq{num_seq}/left'.format(num_seq=i)
            output = output_path + 'train/seq{num_seq}/left/'.format(num_seq=i)
        elif (i <= 15):  # validation
            newpath = output_path + 'validation/seq{num_seq}/left'.format(num_seq=i)
            output = output_path + 'validation/seq{num_seq}/left/'.format(num_seq=i)
        else:            # test
            newpath = output_path + 'test/seq{num_seq}/left'.format(num_seq=i)
            output = output_path + 'test/seq{num_seq}/left/'.format(num_seq=i)

        if not os.path.exists(newpath):
            os.makedirs(newpath)

        imgs_input = input_path + 'seq{num_seq}/left/'.format(num_seq=i)

        print('*' * 100)
        print('Analisi Sequenza {n_seq}/17 in corso ...'.format(n_seq=i))
        print("Generazione Nuove Immagini dall'Input:", imgs_input)
        for image in sorted(os.listdir(imgs_input)):
            if (image.endswith('png')):  # Controlla che si tratti di un'immagine
                img = cv.imread(imgs_input + '{img_name}'.format(img_name=image), cv.IMREAD_UNCHANGED)
                crop_img = img[60:-20, :, :]  # Ritaglio l'immagine

                height = crop_img.shape[0]
                width = crop_img.shape[1]
                new_width = int(width * SCALE_PERCENT)
                new_height = int(height * SCALE_PERCENT)
                dim = (new_width, new_height)

                resized = cv.resize(crop_img, dim, interpolation=cv.INTER_AREA)

                print('Sequenza seq{n_seq}, Image {img} Saving'.format(n_seq=i, img=image))

                cv.imwrite(os.path.join(output, image), resized)
        print('Output Nuove Immagini:', newpath)
        print('*' * 100)


def adjustDepths(input_path='Sequenze-Depth-RGB/', output_path='Dataset/IMG/'):
    for i in range(1, 18):  # Abbiamo 17 {1,2,...,17} sequenze di video

        if (i <= 12):    # train
            newpath = output_path + 'train/seq{num_seq}/depth'.format(num_seq=i)
            output = output_path + 'train/seq{num_seq}/depth/'.format(num_seq=i)
        elif (i <= 15):  # validation
            newpath = output_path + 'validation/seq{num_seq}/depth'.format(num_seq=i)
            output = output_path + 'validation/seq{num_seq}/depth/'.format(num_seq=i)
        else:            # test
            newpath = output_path + 'test/seq{num_seq}/depth'.format(num_seq=i)
            output = output_path + 'test/seq{num_seq}/depth/'.format(num_seq=i)

        if not os.path.exists(newpath):
            os.makedirs(newpath)

        depths_input = input_path + 'seq{num_seq}/depth/'.format(num_seq=i)

        print('*' * 100)
        print('Analisi Sequenza {n_seq}/17 in corso ...'.format(n_seq=i))
        print("Generazione Nuove Immagini dall'Input:", depths_input)
        for image in sorted(os.listdir(depths_input)):
            if (image.endswith('png')):  # Controlla che si tratti di un'immagine
                img = cv.imread(depths_input + '{img_name}'.format(img_name=image), cv.IMREAD_UNCHANGED)
                crop_img = img[60:-20, :]  # Ritaglio l'imagine

                height = crop_img.shape[0]
                width = crop_img.shape[1]
                new_width = int(width * SCALE_PERCENT)
                new_height = int(height * SCALE_PERCENT)
                dim = (new_width, new_height)

                resized = cv.resize(crop_img, dim, interpolation=cv.INTER_AREA)

                print('Sequenza seq{n_seq}, Image {img} Saving'.format(n_seq=i, img=image))

                cv.imwrite(os.path.join(output, image), resized)
        print('Output Nuove Immagini:', newpath)
        print('*' * 100)


def adaptJsonTrajectoriesToNewImages(input_path='Sequenze-Depth-RGB/', output_path='Dataset/IMG/'):
    K = SCALE_PERCENT
    offset = 60

    for i in range(1, 18):  # Abbiamo 17 {1,2,...,17} sequenze di video
        dict = {}
        input = input_path + 'seq{num_seq}/trajectories.json'.format(num_seq=i)

        print('*' * 100)
        print('Analisi Sequenza {n_seq}/17 in corso ...'.format(n_seq=i))
        print("Generazione Nuove Traiettorie dall'Input:", input)

        with open(input) as json_file:
            data = json.load(json_file)

            for elem in data:
                x_curr, y_curr = data[elem]['object_0']['present']
                curr_pos = [round(x_curr * K), round((y_curr - offset) * K)]

                future_positions = []
                for pos in data[elem]['object_0']['future']:
                    future_positions.append([round(pos[0] * K), round((pos[1] - offset) * K)])
                dict.update({elem: {"object_0": {"present": curr_pos, "future": future_positions}}})

        if (i <= 12):    # train
            newpath = output_path + 'train/seq{num_seq}'.format(num_seq=i)
            output = output_path + 'train/seq{num_seq}/trajectories.json'.format(num_seq=i)
        elif (i <= 15):  # validation
            newpath = output_path + 'validation/seq{num_seq}'.format(num_seq=i)
            output = output_path + 'validation/seq{num_seq}/trajectories.json'.format(num_seq=i)
        else:            # test
            newpath = output_path + 'test/seq{num_seq}'.format(num_seq=i)
            output = output_path + 'test/seq{num_seq}/trajectories.json'.format(num_seq=i)

        if not os.path.exists(newpath):
            os.makedirs(newpath)

        print('Output Nuove Traiettoire:', output)
        print('*' * 100)
        with open(output, 'w') as json_file:
            json_file.write(str(dict).replace("\'", "\""))


def genetateDatasetCSV():
    num_of_points = 30

    csvData = []
    csvData.append(['image', 'depth', 'trajectory'])

    for i in range(1, 18):  # Abbiamo 17 {1,2,...,17} sequenze di video

        if (i <= 12):    # train
            type_folder = 'train'
        elif (i <= 15):  # validation
            type_folder = 'validation'
        else:            # test
            type_folder = 'test'

        CSV_PATH = 'CSV/{folder}_driving_log.csv'.format(folder=type_folder)

        path = 'Dataset/IMG/{folder}/seq{num_seq}/'.format(folder=type_folder, num_seq=i)
        imgs_path = path + 'left/'
        trajectories_path = path + 'trajectories.json'.format(num_seq=i)

        with open(trajectories_path) as json_file:
            data = json.load(json_file)
        if (len(data) != 0):
            print('*' * 100)
            print('Generazione Dataset per Sequenza {num_seq}'.format(num_seq=i))
            print('Input Path:', imgs_path)
            print('Output Path:', CSV_PATH)
            print('*' * 100)
            for image_name in sorted(os.listdir(imgs_path)):
                is_image = True if image_name.endswith('.png') else False
                if (is_image == True and int(image_name[4:-4]) > 249):  # Controlla che si tratti di un'immagine
                    n_img = int(image_name[4:-4]) - 1
                    frame = 'frame_{n}'.format(n='0' * (6 - len(str(n_img))) + str(n_img))

                    coorFuture = data[frame]["object_0"]["future"]

                    # img_path = 'IMG/{folder}/seq{num_seq}/left/'.format(folder=type_folder, num_seq=i)
                    # depth_path = 'IMG/{folder}/seq{num_seq}/depth/'.format(folder=type_folder, num_seq=i)

                    # img_path = input_path + 'seq{num_seq}/left/'.format(folder=type_folder, num_seq=i)
                    # depth_path = input_path + 'IMG/{folder}/seq{num_seq}/depth/'.format(folder=type_folder, num_seq=i)

                    depth_path = path + 'depth/'
                    depth_name = 'depth' + image_name[4:]

                    if (len(coorFuture) >= num_of_points):
                        coords = ''
                        for elem in coorFuture[0:num_of_points]:
                            coords = coords + str(elem[0]) + ',' + str(elem[1]) + ':'
                        coords = coords[:-1]
                        csvData.append([imgs_path + image_name, depth_path + depth_name, coords])
        else:
            print('File Traiettorie Sequenza {num} VUOTO!!!'.format(num=i))

        if (i == 12):     # train
            write_csv(CSV_PATH, csvData)
            csvData = []
            csvData.append(['image', 'depth', 'trajectory'])
        elif (i == 15):   # validation
            write_csv(CSV_PATH, csvData)
            csvData = []
            csvData.append(['image', 'depth', 'trajectory'])
        elif (i == 17):   # test
            write_csv(CSV_PATH, csvData)
            csvData = []
    # print(csvData)


def write_csv(CSV_PATH, csvData):
    with open(CSV_PATH, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()


def flipImg(img, traj):
    flip_img = cv.flip(img, 1)
    IMAGE_WIDTH = img.shape[1]
    flip_traj = [[IMAGE_WIDTH - 1 - x, y] for x, y in traj]
    flip_traj = np.array(flip_traj)
    return flip_img, flip_traj


def showTrajectoryOnImages(images_path, trajectories_path, offset = 0):

    '''
    ***____*** ASSUMO CHE offset != 0 ----IMPLICHI----> DI CONSIDERARE LE IMMAGINI A GRANDEZZA NATURALE ***____***
    '''

    num_of_points = 30
    start_from = 249

    images = [img for i, img in enumerate(sorted(os.listdir(images_path))) if img.endswith("png")]
    images[0:start_from + 1] = []  # ----> NB: La predizione viene eseguita a partire dalla 252-esima immagine

    with open(trajectories_path) as json_file:
        data = json.load(json_file)

    print('*' * 100)
    print('Images Length: ', len(images))
    print('Trajectories Length: ', len(data) - (start_from + 1))
    print('*' * 100)

    for image in images:
        img = cv.imread(os.path.join(images_path, image))
        img = img[60:-25, :, :] if offset != 0 else img
        n_img = int(image[-10:-4]) - 1
        frame = 'frame_{n}'.format(n='0' * (6 - len(str(n_img))) + str(n_img))

        x, y = data[frame]["object_0"]["present"]  # (x, y) coordinate punto corrente
        cv.circle(img, (x, y  - offset), 3, (0, 255, 0), 1)

        coorFuture = data[frame]["object_0"]["future"]
        coorFuture = coorFuture[0:num_of_points] if len(coorFuture) >= num_of_points else coorFuture # Vengono mostrati solo i primi 50 punti

        # img, coorFuture = flipImg(img, coorFuture) # FLIP dell'immagine

        for item in coorFuture:
            y = int(item[1]) - offset
            x = int(item[0])
            img[y - 1:y + 1, x - 1:x + 1] = [255, 0, 0]

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, 'Real Traj', (10, 25), font, 0.35, (255, 0, 0), 1, cv.LINE_AA)
        cv.putText(img, 'Curr Pos', (10, 45), font, 0.35, (0, 255, 0), 1, cv.LINE_AA)

        cv.imshow("'ESC' to exit", img)

        k = cv.waitKey(40) # (1000 ms : 25 frame) = 40 ms/frame
        if (k == 27):  # Se viene premuto il tasto " ESC" termina la riproduzione del video
            break
    cv.destroyAllWindows()  # Deallocating memories taken for window creation


def showPredictedTrajectoryOnImages(test_path, trajectories_file, K = 1):
    '''
    ***____*** ASSUMO CHE K != 1 ----IMPLICHI----> DI CONSIDERARE LE IMMAGINI A GRANDEZZA NATURALE ***____***
    '''

    start_from = 249

    images = [img for i, img in enumerate(sorted(os.listdir(test_path))) if img.endswith("png")]
    images[0:start_from + 1] = []
    with open(trajectories_file) as json_file:
        data = json.load(json_file)
        data = np.array(data)

    print('*' * 100)
    print('Images Length: ', len(images))
    print('Trajectories Length: ', data.shape)
    print('*' * 100)

    for i, image in enumerate(images):
        img = cv.imread(os.path.join(test_path, image))
        img = img[60:-25, :, :] if K != 1 else img

        coorFuture = data[i] * K
        # print(coorFuture)

        for item in coorFuture:
            y = int(item[1])
            x = int(item[0])
            img[y - 1:y + 1, x - 1:x + 1] = [255, 0, 0]

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, 'Pred Traj', (10, 25), font, 0.35, (255, 0, 0), 1, cv.LINE_AA)

        cv.imshow("'ESC' to exit", img)
        k = cv.waitKey(40)  # (1000 ms : 25 frame) = 40 ms/frame
        if (k == 27):  # Se viene premuto il tasto " ESC" termina la riproduzione del video
            break
    cv.destroyAllWindows()  # Deallocating memories taken for window creation


def save_video_with_trajectory(video_name, images_path, pred_trajs, real_trajs, type, output_path):

    '''
           {type = 0 -> PUNTI; type = 1 -> LINEE}
    '''

    K = 8 # To adapt predicted points to the original images sizes
    offset = 60
    start_from = 249
    num_of_points = 30

    images = [img for i, img in enumerate(sorted(os.listdir(images_path)))
              if img.endswith("png")]

    images[0:start_from + 1] = []  # ----> NB: La predizione viene eseguita a partire dalla 251-esima immagine

    with open(pred_trajs) as json_file:
        pred_data = json.load(json_file)
        pred_data = np.array(pred_data)

    with open(real_trajs) as json_file:
        real_data = json.load(json_file)

    new_data = []
    centers = []
    for i, key in enumerate(real_data.keys()):
        if (i > start_from):
            trajectory = real_data[key]["object_0"]["future"]
            elem = trajectory[0:num_of_points] if len(trajectory) >= num_of_points else trajectory
            new_data.append(elem)
            centers.append(real_data[key]["object_0"]["present"])

    centers = np.array(centers)
    real_data = np.array(new_data)

    print('*' * 100)
    print('Images Length: ', len(images))
    print('Pred Trajectory Length:', len(pred_data))
    print('Real Trajectory Length:', len(real_data))
    print('\n', '*** Waiting... Video Creation ***', '\n')
    print('*' * 100)

    if type == 0:
        output_path = output_path + 'PUNTI/'
    elif type == 1:
        output_path = output_path + 'LINEE/'
    else:
        print('*'*10, 'ERRORE!', '*'*10)
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter(output_path + video_name, fourcc, 24, (1280, 635), 1)

    for ind, image in enumerate(images):
        img = cv.imread(os.path.join(images_path, image))
        img = img[60:-25, :, :]

        pred_coorFuture = pred_data[ind] * K
        real_coorFuture = real_data[ind]

        x, y = centers[ind]  # (x, y) coordinate punto corrente
        cv.circle(img, (x, y - offset), 3, (0, 255, 0), 2)  # 2 -> 1

        for i, item in enumerate(pred_coorFuture):
            x = int(item[0])
            y = int(item[1])
            if (type == 0):  # Disegno i punti
                img[y - 2:y + 2, x - 2:x + 2] = [0, 0, 255]

            elif (type == 1):  # Disegno le linee
                if (i < (len(pred_coorFuture) - 1)):
                    next_x = int(pred_coorFuture[i + 1][0])
                    next_y = int(pred_coorFuture[i + 1][1])
                    cv.line(img, (x, y), (next_x, next_y), (0, 0, 255), 2)

        for i, item in enumerate(real_coorFuture):
            x = int(item[0])
            y = int(item[1]) - offset
            if (type == 0):  # Disegno i punti
                img[y - 2:y + 2, x - 2:x + 2] = [0, 255, 255]  # (255, 0, 0) -> ROSSO

            elif (type == 1):  # Disegno le linee
                if (i < (len(real_coorFuture) - 1)):
                    next_x = int(real_coorFuture[i + 1][0])
                    next_y = int(real_coorFuture[i + 1][1] - offset)
                    cv.line(img, (x, y), (next_x, next_y), (0, 255, 255), 2)

        font = cv.FONT_HERSHEY_SIMPLEX

        cv.putText(img, 'Current Car Position', (100, 500), font, 1, (0, 255, 0), 2, cv.LINE_AA)  # 600 -> 200
        cv.putText(img, 'Predicted Trajectory', (100, 550), font, 1, (0, 0, 255), 2, cv.LINE_AA)  # 650 -> 150
        cv.putText(img, 'Real Trajectory', (100, 600), font, 1, (0, 255, 255), 2, cv.LINE_AA)  # (255, 0, 0) -> ROSSO, 700 -> 100

        video.write(img)

        '''
        if(ind == 550):
            cv.imwrite('16_550.png', img)
        if (ind == 750):
            cv.imwrite('17_750.png', img)
        '''

    cv.destroyAllWindows()  # Deallocating memories taken for window creation
    video.release()


def compute_MAE_and_MSE(pred_trajs, real_trajs, point=30, multiple=False, maePerFrame=False):
    '''********'''
    '''
            multiple = False -> Consider only the element in (point - 1) position
            multiple = True  -> Consider the first 'point' elements (i.e. if point = 5, the first 5th elements are analyzed)
            maePerFrame = True -> Allow to save the plot regarding both MSE for single frames and global MSE: it can be True only if multiple = True
    '''
    '''********'''

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    K = 8  # To adapt predicted points to the original images sizes
    offset = 60
    start_from = 249

    n_seq = real_trajs.split('/')[1]
    n_net = pred_trajs.split('/')[1]
    n_epoch = pred_trajs[-8:-5]

    with open(pred_trajs) as json_file:
        pred_data = json.load(json_file)

    with open(real_trajs) as json_file:
        real_data = json.load(json_file)

    new_data = []

    if multiple == False:
        maePerFrame = False
        pred_data = np.array(pred_data)[:, point - 1].reshape((len(pred_data), 2)) * K
        for i, key in enumerate(real_data.keys()):
            if (i > start_from):
                trajectory = real_data[key]["object_0"]["future"]
                if len(trajectory) >= point:
                    elem = trajectory[point - 1]  # Salvo solamente il 30° punto
                    new_data.append([elem[0], elem[1] - offset])
        real_data = np.reshape(new_data, (len(new_data), 2))
    else:
        pred_data = np.array(pred_data)[:, 0:point].reshape((point * len(pred_data), 2)) * K
        for i, key in enumerate(real_data.keys()):
            if (i > start_from):
                trajectory = real_data[key]["object_0"]["future"]
                if len(trajectory) >= point:
                    elements = trajectory[0:point]
                    new_data.append(elements)

        real_data = np.reshape(new_data, (point * len(new_data), 2))

        for i, _ in enumerate(real_data):
            real_data[i][1] -= offset

    pred_data = pred_data[0:len(real_data)]  # Devo fare in modo che i vettori abbiano la solita lunghezza
    mae = mean_absolute_error(real_data, pred_data)
    mse = mean_squared_error(real_data, pred_data)

    print('^^^^^^^^^^^^^^^^^^^^^^^'*2)
    print('Sequenza: {seq}, Rete: {net} , Epoca: {ep}'.format(seq=n_seq, net=n_net, ep=n_epoch), '\n')

    print('Shape Pred Data:', pred_data.shape)
    print('Shape Real Data:', real_data.shape)
    print()
    print("'Metrics on {n} points'".format(n=point)) if multiple else print("'Metrics on {n}° point'".format(n=point))
    print('mae:', mae)
    print('mse:', mse)
    print('^^^^^^^^^^^^^^^^^^^^^^^'*2)

    if (maePerFrame):

        results_path = 'PLOTS/{net}/{seq}/'.format(net=n_net, seq=n_seq)
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        pred_data_split = np.array_split(pred_data, len(pred_data) / point)  # Splitto punti in frame
        real_data_split = np.array_split(real_data, len(real_data) / point)
        maeV = [mae for _ in
                range(len(real_data_split))]  # Lista avente tutti gli elementi uguali a 'mae': SERVE PER FARE PLOT
        statistics = [mean_absolute_error(pred_data_split[i], real_data_split[i]) for i in
                      range(len(real_data_split))]  # Lista in cui salvo mae per ogni frame

        if (point == 1):
            plt.title('MAE on the first Point of each Frame inside the Sequence {seq}'.format(seq=n_seq[-2:]))
        elif (point > 1):
            plt.title('MAE on the first {n} Points of each Frame inside the Sequence {seq}'.format(n=point, seq=n_seq[-2:]))

        plt.plot(statistics, label='MAE for Frame', color = 'green')
        plt.plot(maeV, label='Global MAE', color = 'red')
        plt.legend(loc='upper left')
        plt.ylabel('MAE')
        plt.xlabel('Frame')
        plt.savefig(results_path + 'MAEon{n}points.png'.format(n=point))  # For saving
        plt.show()

    return mae, mse

def compute_distances():

    K = 8  # To adapt predicted points to the original images sizes
    offset = 60
    start_from = 249

    RGB = []
    RGBD = []

    list_pred_trajs = ['MAE/100/#3/16.json', 'MAE/100/#3/17.json', 'MAE/100/#4/16.json', 'MAE/100/#4/17.json'] # Predicted Trajectories
    list_real_trajs = ['MAE/real/16.json', 'MAE/real/17.json'] # Real Trajectories

    path = 'Sequenze-Depth-RGB/seq16/depth/'
    depths = [depth for i, depth in enumerate(sorted(os.listdir(path))) if depth.endswith("png")]
    depths[0:start_from + 1] = []  # ----> NB: La predizione viene eseguita a partire dalla 251-esima immagine
    depths16 = [cv.imread(path + depth, -1)[60:-20] for depth in depths]

    path = 'Sequenze-Depth-RGB/seq17/depth/'
    depths = [depth for i, depth in enumerate(sorted(os.listdir(path))) if depth.endswith("png")]
    depths[0:start_from + 1] = []  # ----> NB: La predizione viene eseguita a partire dalla 251-esima immagine
    depths17 = [cv.imread(path + depth, -1)[60:-20] for depth in depths]

    points = [i for i in range(1, 31)]
    # points = [20]
    for point in points:
        distances_vect = []
        for index, pred_trajs in enumerate(list_pred_trajs):

            real_trajs = list_real_trajs[0] if(pred_trajs.split('/')[3][0:2] == '16') else list_real_trajs[1]
            depth = depths16[index] if (pred_trajs.split('/')[3][0:2] == '16') else depths17[index]

            # depth[depth > 10000] = 10000

            with open(pred_trajs) as json_file:
                pred_data = json.load(json_file)

            with open(real_trajs) as json_file:
                real_data = json.load(json_file)

            new_data = []

            pred_data = np.array(pred_data)[:, point - 1].reshape((len(pred_data), 2)) * K
            for i, key in enumerate(real_data.keys()):
                if (i > start_from):
                    trajectory = real_data[key]["object_0"]["future"]
                    if len(trajectory) >= point:
                        elem = trajectory[point - 1]  # Salvo solamente il 30° punto
                        new_data.append([elem[0], elem[1] - offset])
            real_data = np.reshape(new_data, (len(new_data), 2))

            pred_data = pred_data[0:len(real_data)]  # Devo fare in modo che i vettori abbiano la solita lunghezza

            pred_data_in_meters = []
            real_data_in_meters = []

            for elem in pred_data:
                x = int(elem[0])
                y = int(elem[1])

                value_depth = 0.0
                for j in range(-1, 2):
                    for i in range(-1, 2):
                        value_depth += depth[y - i, x - j]
                value_depth = value_depth / 9

                x_in_m = (x - 1013.93) * value_depth / 1401.64
                y_in_m = (x - 566.906) * value_depth / 1401.64
                z_in_m = value_depth

                pred_data_in_meters.append(np.array([x_in_m, y_in_m, z_in_m]))

            pred_data_in_meters = np.array(pred_data_in_meters)

            for elem in real_data:
                x = int(elem[0])
                y = int(elem[1])

                value_depth = 0.0
                for j in range(-1, 2):
                    for i in range(-1, 2):
                        value_depth += depth[y - i, x - j]
                value_depth = value_depth / 9

                x_in_m = (x - 1013.93) * value_depth / 1401.64
                y_in_m = (x - 566.906) * value_depth / 1401.64
                z_in_m = value_depth

                real_data_in_meters.append(np.array([x_in_m, y_in_m, z_in_m]))

            real_data_in_meters = np.array(real_data_in_meters)

            distance = 0
            for i, elem in enumerate(real_data_in_meters):
                distance += pow(elem[0] - pred_data_in_meters[i][0], 2) + pow(elem[1] - pred_data_in_meters[i][1], 2) + pow(elem[2] - pred_data_in_meters[i][2], 2)

            distances_vect.append(pow(distance / (3 * len(pred_data_in_meters)), 1 / 2))

        distanceRGB = ((distances_vect[0] + distances_vect[1]) / 2) / 10
        distanceRGBD = ((distances_vect[2] + distances_vect[3]) / 2) / 10

        print('*'*100)
        print('Rete RGB,  punto {pt}:'.format(pt=point), distanceRGB, 'centimetri')
        print('Rete RGBD, punto {pt}:'.format(pt=point), distanceRGBD, 'centimetri')
        print('*' * 100)

        RGB.append(distanceRGB)
        RGBD.append(distanceRGBD)


    #''' # Per PLOTTARE i RISULTATI
    plt.plot([x for x in range(1, 31)], RGB, 'go--', label='Distance prediction - ground truth')  # 'bo--'  ->  {b = blue, o = cerchio, -- = linea tratteggiata}
    plt.title('RGB Network')
    plt.legend(loc='upper left')
    plt.ylabel('Distance [cm]')
    plt.xlabel('Point')
    plt.savefig('MAE/distancesRGB.png')  # For saving
    plt.show()

    plt.plot([x for x in range(1, 31)], RGBD, 'go--', label='Distance prediction - ground truth')  # 'bo--'  ->  {b = blue, o = cerchio, -- = linea tratteggiata}
    plt.title('RGBD Network')
    plt.legend(loc='upper left')
    plt.ylabel('Distance [cm]')
    plt.xlabel('Point')
    plt.savefig('MAE/distancesRGBD.png')  # For saving
    plt.show()
    #'''

def draw_plot(dict, points):
    print('+' * 35)
    for elem in dict:
        for key in elem.keys():

            print('Salvataggio dei Plot in corso ... {k}'.format(k = key))

            n_seq = key.split('/')[2][-2:]

            results_path = key

            if not os.path.exists(results_path):
                os.makedirs(results_path)

            plt.plot(points, elem[key], 'bo--', label='MAE of Sequence {n}'.format(n=n_seq)) # 'bo--'  ->  {b = blue, o = cerchio, -- = linea tratteggiata}

            plt.legend(loc='upper left')
            plt.ylabel('MAE')
            plt.xlabel('Points')
            plt.savefig(results_path + 'MAEtotalSequence{n}.png'.format(n=n_seq[-2:]))  # For saving
            plt.show()
    print('+' * 35)

def checkImagesCorrectness():

    print('-' * 1000)
    print('***************** checkImagesCorrectness() *****************')
    for n_seq in range(1, 18):
        if (n_seq <= 12):    # train
            type = 'train'
        elif (n_seq <= 15):  # validation
            type = 'validation'
        else:                # test
            type = 'test'

        path = 'Dataset/IMG/{type}/seq{n}/left/'.format(type=type, n=n_seq)

        for image in sorted(os.listdir(path)):

            depth_path = 'Dataset/IMG/{type}/seq{n}/depth/'.format(type=type, n=n_seq)

            is_image = True if image.endswith("png") else False
            if (is_image):
                img_path = path + image
                depth = 'depth' + image[4:]
                depth_path = depth_path + depth

                img_exists = os.path.exists(img_path)
                depth_exists = os.path.exists(depth_path)

                problem = True
                if img_exists * depth_exists == 1:
                    problem = False

                print('Sequence {n}, {type}:'.format(type=type, n=n_seq), image, depth, 'Problem ->', problem)
    print('-' * 100)


if __name__ == "__main__":
    print()

    '''_____****_____FUNZIONI PER GESTIRE LE IMMAGINI E LE TRAIETTORIE DEL DATASET_____****_____'''
    # adjustImages()
    # adjustDepths()
    # checkImagesCorrectness() # Scrive {Sequenza, image_name, depth_name, problema = {True, False}}, se problema è True => problema immagine-depth corrispondente
    # adaptJsonTrajectoriesToNewImages()
    # genetateDatasetCSV()
    '''_____****_____^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^_____****_____'''




    '''_____****_____FUNZIONI PER VISUALIZZARE LE MODIFICHE E I RISULTATI SULLE IMMAGINI DEL DATASET_____****_____'''
    '''___________________________________________________________________________________________________________'''
    '''
            RETE '#1' [dropout(0.5), 256, 128, 60)] -> RGB,   lr/2 ogni 50 epoche, senza Data Augmentation
            RETE '#2' [dropout(0.5), 256, 128, 60)] -> RGBD,  lr/2 ogni 50 epoche, senza Data Augmentation
            RETE '#3' [dropout(0.5), 256, 128, 60)] -> RGB,   lr/2 ogni 50 epoche, con   Data Augmentation
            RETE '#4' [dropout(0.5), 256, 128, 60)] -> RGBD,  lr/2 ogni 50 epoche, con   Data Augmentation  
    '''
    '''___________________________________________________________________________________________________________'''




    '''___________________________________________________________________________________________________________'''
    '''***********************************************************************************************************'''
    ''' ** compute_distances() ** ''' # Funzione per calcolare le distanze in centimetri tra le traiettorie reali e quelle predette
    '''***********************************************************************************************************'''
    '''
    compute_distances()
    '''
    '''___________________________________________________________________________________________________________'''




    '''___________________________________________________________________________________________________________'''
    '''***********************************************************************************************************'''
    ''' ** PARAMETRI PER LE DUE FUNZIONI SUCCESSIVE ** '''
    '''***********************************************************************************************************'''
    sequences = ['16', '17']
    networks = ['1', '2', '3', '4']
    epochs = ['050', '070', '100', '150', '200']

    sequences = ['16', '17']
    networks = ['3', '4']
    epochs = ['050', '100', '200']
    '''___________________________________________________________________________________________________________'''




    '''___________________________________________________________________________________________________________'''
    '''***********************************************************************************************************'''
    ''' ** save_video_with_trajectory(video_name, imgs_path, pred_trajs, real_trajs) ** '''
    '''***********************************************************************************************************'''
    '''
    for n_net in networks:
        for n_seq in sequences:
            for n_epoch in epochs:
                print('^^^^^^^^^^^^^^^^^^^^^^^')
                print('NETWORK #{net} EP{epoch}'.format(net=n_net, epoch = n_epoch), 'Video Sequence:', n_seq)
                print('^^^^^^^^^^^^^^^^^^^^^^^')
                
                realTrajs_json = 'Sequenze-Depth-RGB/seq{n}/trajectories.json'.format(n=n_seq)
                predTrajs_json = 'Predictions/#{net}/#{net} prediction_seq{n}_ep{epoch}.json'.format(net=n_net, n=n_seq, epoch=n_epoch)
                input = 'Sequenze-Depth-RGB/seq{n_seq}/left/'.format(n_seq=n_seq) # Percorso delle immagini originali

                output = '*VIDEO*/RETE #{net}/30P_{ep}EP/'.format(net=n_net, ep=n_epoch) # Percorso dove voglio che vengano salvati i dati

                save_video_with_trajectory(video_name='Seq{n}.mp4'.format(net=n_net, n=n_seq, epoch=n_epoch), #.avi
                                           images_path=input,
                                           pred_trajs=predTrajs_json,
                                           real_trajs=realTrajs_json,
                                           type=1, # {type = 0 -> PUNTI; type = 1 -> LINEE}
                                           output_path=output)
    '''
    '''___________________________________________________________________________________________________________'''




    '''___________________________________________________________________________________________________________'''
    '''***********************************************************************************************************'''
    ''' *** compute_MAE_and_MSE(pred_trajs, real_trajs) *** '''
    '''***********************************************************************************************************'''
    '''
    points = [1, 5, 10, 15, 20, 25, 30]

    dict = []

    for n_seq in sequences:
        for n_net in networks:
            for n_epoch in epochs:
                realTrajs_json = 'Sequenze-Depth-RGB/seq{n}/trajectories.json'.format(n=n_seq)
                predTrajs_json = 'Predictions/#{net}/#{net} prediction_seq{n}_ep{epoch}.json'.format(net=n_net, n=n_seq, epoch=n_epoch)

                list_of_mae = []

                for point in points:
                    mae, mse = compute_MAE_and_MSE(pred_trajs=predTrajs_json,
                                                   real_trajs=realTrajs_json,
                                                   point=point,
                                                   multiple=True,
                                                   maePerFrame=False)  # True
                    list_of_mae.append(mae)

                path = 'PLOTS/#{net}/seq{seq}/'.format(net=n_net, seq=n_seq) # NB: Mettere come chiave il percorso dove si vuole salvare il plot
                dict.append({path: list_of_mae})  # Per ogni sequenza e per ogni rete salvo i vari mae

    draw_plot(dict=dict, points=points) # Salvo il plot per ogni rete per ogni sequenza dei corrispettivi mae
    '''
    '''___________________________________________________________________________________________________________'''




    '''___________________________________________________________________________________________________________'''
    '''***********************************************************************************************************'''
    ''' ** PARAMETRI PER LE DUE FUNZIONI SUCCESSIVE ** '''
    '''***********************************************************************************************************'''
    n_net = str(3)      # {1, 2, 3, 4}
    n_seq = str(17)     # Sequenza 16 dei Video
    n_epoch = '100'     # Epoca dei pesi della rete
    '''___________________________________________________________________________________________________________'''




    '''___________________________________________________________________________________________________________'''
    '''***********************************************************************************************************'''
    ''' *** showTrajectoryOnImages(images_path, trajectories_path, offset = 0) *** '''
    '''***********************************************************************************************************'''
    '''
    input = 'Dataset/IMG/test/seq{n}/left/'.format(n=n_seq)             # IMMAGINI RIDOTTE
    realTrajs_json = 'Dataset/IMG/test/seq{n}/trajectories.json'.format(n=n_seq)
    showTrajectoryOnImages(images_path=input,
                           trajectories_path=realTrajs_json,
                           offset=0)
    '''
    '''
    input = 'Sequenze-Depth-RGB/seq{n}/left/'.format(n=n_seq)           # IMMAGINI ORIGINALI
    realTrajs_json = 'Sequenze-Depth-RGB/seq{n}/trajectories.json'.format(n=n_seq)
    showTrajectoryOnImages(images_path=input,
                           trajectories_path=realTrajs_json,
                           offset=60)
    '''
    '''___________________________________________________________________________________________________________'''




    '''___________________________________________________________________________________________________________'''
    '''***********************************************************************************************************'''
    ''' *** showPredictedTrajectoryOnImages(test_path, trajectories_file, K = 1) *** '''
    '''***********************************************************************************************************'''
    '''
    input = 'Dataset/IMG/test/seq{n}/left/'.format(n=n_seq)             # IMMAGINI RIDOTTE
    predTrajs_json = 'Predictions/#{net}/#{net} prediction_seq{n}_ep{epoch}.json'.format(net=n_net, n=n_seq, epoch=n_epoch)
    showPredictedTrajectoryOnImages(input, predTrajs_json, K=1)
    '''
    '''
    input = 'Sequenze-Depth-RGB/seq{n}/left/'.format(n=n_seq)           # IMMAGINI ORIGINALI
    predTrajs_json = 'Predictions/#{net}/#{net} prediction_seq{n}_ep{epoch}.json'.format(net=n_net, n=n_seq, epoch=n_epoch)
    showPredictedTrajectoryOnImages(input, predTrajs_json, K=8)
    '''
    '''___________________________________________________________________________________________________________'''
    '''_____****_____^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^_____****_____'''