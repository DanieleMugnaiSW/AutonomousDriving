# AutonomousDriving
#### main.py
Images Preprocessing (to improve network training):
	 	 adjustImages(input_path, output_path) (1280x720x3 -> 160x80x3)
	 	 adjustDepths(input_path, output_path) (1280x720x1 -> 160x80x1)
	 	 adaptJsonTrajectoriesToNewImages(input_path, output_path)
		

	 	 input_path: 'Sequenze-Depth-RGB/'
	 	 output_path: 'Dataset/IMG/'

CSV Files Generation (for loading data during the training):
	 	 generateDatasetCSV()
	 

	 	 INPUT: 'Dataset/IMG/'
	 	 OUTPUT: 'CSV/'
To Check the Network Predictions:
	 	 save_video_with_trajectory(video_name, images_path, pred_trajs, real_trajs, type, output_path)
	 	 video_name: how I want named video
	 	 images_path: path of the images where I did the prediction
	 	 pred_trajs: Json containing predicted points
	 	 real_trajs: Json containing real points
	 	 type: {0, 1}; 0 -> trajectory as points, 1 -> trajectory as lines
	 	 output_path: where I want to save video
	 

To Evaluate Correctness of Predictions:
	 	 compute_distances()
		

	 	 These lists are present inside the function:
	 	 	 list_pred_trajs: about predicted trajectories
 	 	 	 list_real_trajs: about real trajectories
	 	 	 NB: Each element of these list is a Json path


#### CNN.ipynb
##### Network Training:
1. Open your Google Drive
2. Add 'Keras Network’ folder to your Drive
3. - ‘Keras Network‘ folder contains the following folders:
	 	 	 * ‘Dataset’ -> ‘IMG‘ //Folder containing images
	 	 	 * ‘CSV' //Folder containing CSV files
	 	 	 * 'Saved Model‘ //Create this folder to save your model
4. Mount your Drive and position yourself in the ‘Keras Network/‘ folder
5. Follow the indications present in the ’CNN.ipynb' Notebook
