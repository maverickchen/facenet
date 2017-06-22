import fn_classifier_fn as fn
import tensorflow as tf
import os
# sess = classify.init_session()
# print(sess)
# close_session(sess)
# print("Made it")


with tf.Graph().as_default():
	with tf.Session() as sess:
		fn.init_session_model()
		data_dir = '~/datasets/test/test_mtcnnpy_160'
		img_path = '~/datasets/test/test_mtcnnpy_160/Maverick_Chen/Photo on 6-16-17 at 9.56 AM.png'
		img_path_exp = os.path.expanduser(img_path)
		cam_classifier_path = '~/models/cam_classifier.pkl'
		fn.train(sess,data_dir,cam_classifier_path)
		fn.classify(sess,img_path_exp,cam_classifier_path)

