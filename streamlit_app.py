import os
import streamlit as st
import numpy as np
from keras.preprocessing import image


def file_selector(folder_path='.'):
	filenames = os.listdir(folder_path)
	selected_filename = st.selectbox('Select a file', filenames)
	return os.path.join(folder_path, selected_filename)

def main():
	st.title('Image Segmentation')

	#Select an image
	st.write("Select an image to obtain the predicted mask")
	if st.checkbox("Click to select"):
		folder_path = st.text_input('Test Images folder', 'streamlit_images')
		filenames = os.listdir(folder_path)
		filename = file_selector(folder_path=folder_path)

		img_name = filename[filename.index('/'):]
		st.write('You selected `%s`' % img_name[1:])
		st.write('Choose a different image using the drop down')

		img_num = img_name[6]
		pred_mask_img = os.path.join(folder_path, "..\\mask_images")
		pred_mask_img = pred_mask_img+"\\pred_mask"+str(img_num)+".png"

		true_mask_img =	 os.path.join(folder_path, "..\\mask_images")
		true_mask_img = true_mask_img+"\\true_mask"+str(img_num)+".png"

		img = image.load_img(filename, target_size=(224,224,3))
		x = image.img_to_array(img)
		x = np.expand_dims(img, axis=0)
		st.image(img, caption="Selected image")

		img = image.load_img(pred_mask_img, target_size=(224,224,3))
		x = image.img_to_array(img)
		x = np.expand_dims(img, axis=0)
		st.image(img, caption="The predicted mask")

		img = image.load_img(true_mask_img, target_size=(224,224,3))
		x = image.img_to_array(img)
		x = np.expand_dims(img, axis=0)
		st.image(img, caption="The true mask")


if __name__ == '__main__':
	main()


