import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil
import cv2
import caffe
from caffe.proto import caffe_pb2
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage


REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '\\..\\..')
UPLOAD_FOLDER = 'D:\\web_AI\\Upload\\'
UPLOAD_FOLDER_eritrosit = 'D:\\web_AI\\Upload\\Eritrosit\\'
UPLOAD_FOLDER_lenfosit = 'D:\\web_AI\\Upload\\Lenfosit\\'
UPLOAD_FOLDER_monosit = 'D:\\web_AI\\Upload\\Monosit\\'
UPLOAD_FOLDER_trombosit = 'D:\\web_AI\\Upload\\Trombosit\\'
UPLOAD_FOLDER_notrofil = 'D:\\web_AI\\Upload\\Notrofil\\'
File1 = 'D:\\web_AI\\images\\abc_7898.jpg'
File2 = 'D:\\web_AI\\images\\abc_7900.jpg'
File3 = 'D:\\web_AI\\images\\abc_7909.jpg'
File4 = 'D:\\web_AI\\images\\abc_7910.jpg'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    img1 = exifutil.open_oriented_im(File1)
    img2 = exifutil.open_oriented_im(File2)
    img3 = exifutil.open_oriented_im(File3)
    img4 = exifutil.open_oriented_im(File4)
    return flask.render_template('index.html', has_result=False, imagesrc_1=embed_image_html(img1), imagesrc_2=embed_image_html(img2), imagesrc_3=embed_image_html(img3), imagesrc_4=embed_image_html(img4))

@app.route('/classify_url', methods=['GET'])
def classify_url():
        imageid = flask.request.args.get('imageid', '')
        imagename = ''
        if imageid=='1':
           imagename = File1
        elif imageid=='2':
           imagename = File2
        elif imageid=='3':
           imagename = File3
        elif imageid=='4':
           imagename = File4
        print imagename
        img = classify_file(imagename)
        img1 = exifutil.open_oriented_im(File1)
        img2 = exifutil.open_oriented_im(File2)
        img3 = exifutil.open_oriented_im(File3)
        img4 = exifutil.open_oriented_im(File4)
        return flask.render_template(
        'index.html', has_result=True, result='result',
        imagesrc=embed_image_html(img), imagesrc_1=embed_image_html(img1), imagesrc_2=embed_image_html(img2), imagesrc_3=embed_image_html(img3), imagesrc_4=embed_image_html(img4))

@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        cwd = os.getcwd()
        print cwd
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + werkzeug.secure_filename(imagefile.filename)
        filename_ = str(filename_).replace(':', '_')
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = cv2.imread(filename)
        r_img = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #kernel = np.ones((5,5),np.uint8)

        #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        #sure_bg = cv2.erode(opening,kernel,iterations=3)
        #sure_bg = cv2.erode(opening,kernel,iterations=3)

        criteria = np.uint8(thresh)
        height,width = criteria.shape[:2]
        blank_image = np.zeros((height,width), np.uint8)
        im2, contours, hierarchy = cv2.findContours(criteria,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #cv2.fillPoly(blank_image, pts =[contours], color=(255,255,255))
        cv2.drawContours(blank_image, contours, -1, (255), cv2.FILLED)
        #cv2.imshow('image',blank_image)
        #cv2.waitKey(0)
       

        D = ndimage.distance_transform_edt(blank_image)
        localMax = peak_local_max(D, indices=False, min_distance=10,labels=blank_image)
 
# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=blank_image)
        print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
# loop over the unique labels returned by the Watershed
# algorithm
        c_img = image.copy()
        segment_id = 0
        for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	     if label == 0:
		continue
 
	# otherwise, allocate memory for the label region and draw
	# it on the mask
	     mask = np.zeros(gray.shape, dtype="uint8")
	     mask[labels == label] = 255
 
	# detect contours in the mask and grab the largest one
	     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	     cv2.CHAIN_APPROX_SIMPLE)[-2]
	     c = max(cnts, key=cv2.contourArea)
             x, y, w, h = cv2.boundingRect(c)
             x = x-20
             y = y-20
             w = w+40
             h = h+40
             c_imag = c_img.copy() 
             roi = c_imag[y:y+h, x:x+w,:]
             if len(roi):
             	result = app.clf.classify_image(roi.copy())
             	#cv2.rectangle(image, (x, y), (x+w, y+h), (255),1)
             	time = str(datetime.datetime.now()).replace(' ', '_')
                time = str(time).replace(':', '_')
             	if result[1]=='lenfosit':
                 	cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,0),2)
                        cv2.imwrite(UPLOAD_FOLDER_lenfosit+'lenfosit_' +time +str(segment_id)+'.jpg', roi)
             	elif result[1]=='eritrosit':
                 	cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0),2)
                        cv2.imwrite(UPLOAD_FOLDER_eritrosit+'eritrosit_' +time+str(segment_id)+'.jpg', roi)
             	elif result[1]=='monosit':
                 	cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0),2)
                        cv2.imwrite(UPLOAD_FOLDER_monosit +'monosit_'+time+str(segment_id)+'.jpg', roi)
             	elif result[1]=='notrofil':
                 	cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,255),2)
                        cv2.imwrite(UPLOAD_FOLDER_notrofil +'notrofil_'+time+str(segment_id)+'.jpg', roi)
             	elif result[1]=='trombosit':
                 	cv2.rectangle(image, (x, y), (x+w, y+h), (203,102,255),2)
                        cv2.imwrite(UPLOAD_FOLDER_trombosit +'trombosit_'+time+str(segment_id)+'.jpg', roi)
	# draw a circle enclosing the object
	     #((x, y), r) = cv2.minEnclosingCircle(c)
	     #cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
	     #cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
             #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                segment_id = segment_id +1
        cv2.imwrite(filename+'_AI', image)
        cv2.imwrite(filename, c_img)
        img = exifutil.open_oriented_im(filename+'_AI')


    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    #
    img1 = exifutil.open_oriented_im(File1)
    img2 = exifutil.open_oriented_im(File2)
    img3 = exifutil.open_oriented_im(File3)
    img4 = exifutil.open_oriented_im(File4)
    return flask.render_template('index.html', has_result=True, result=result, imagesrc=embed_image_html(img),imagesrc_1=embed_image_html(img1), imagesrc_2=embed_image_html(img2), imagesrc_3=embed_image_html(img3), imagesrc_4=embed_image_html(img4))
	
def classify_file(img):
    try:
        print 'aaaaaaaaaaaaa'
        image = cv2.imread(img)
        print 'aaaaaaaaaaaaa'
        r_img = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #kernel = np.ones((5,5),np.uint8)

        #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        #sure_bg = cv2.erode(opening,kernel,iterations=3)
        #sure_bg = cv2.erode(opening,kernel,iterations=3)

        criteria = np.uint8(thresh)
        height,width = criteria.shape[:2]
        blank_image = np.zeros((height,width), np.uint8)
        im2, contours, hierarchy = cv2.findContours(criteria,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #cv2.fillPoly(blank_image, pts =[contours], color=(255,255,255))
        cv2.drawContours(blank_image, contours, -1, (255), cv2.FILLED)
        #cv2.imshow('image',blank_image)
        #cv2.waitKey(0)
       

        D = ndimage.distance_transform_edt(blank_image)
        localMax = peak_local_max(D, indices=False, min_distance=10,labels=blank_image)
 
# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=blank_image)
        print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
# loop over the unique labels returned by the Watershed
# algorithm
        c_img = image.copy()
        segment_id = 0
        for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	     if label == 0:
		continue
 
	# otherwise, allocate memory for the label region and draw
	# it on the mask
	     mask = np.zeros(gray.shape, dtype="uint8")
	     mask[labels == label] = 255
 
	# detect contours in the mask and grab the largest one
	     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	     cv2.CHAIN_APPROX_SIMPLE)[-2]
	     c = max(cnts, key=cv2.contourArea)
             x, y, w, h = cv2.boundingRect(c)
             x = x-20
             y = y-20
             w = w+40
             h = h+40
             c_imag = c_img.copy() 
             roi = c_imag[y:y+h, x:x+w,:]
             if len(roi):
             	result = app.clf.classify_image(roi.copy())
             	#cv2.rectangle(image, (x, y), (x+w, y+h), (255),1)
             	time = str(datetime.datetime.now()).replace(' ', '_')
                time = str(time).replace(':', '_')
             	if result[1]=='lenfosit':
                 	cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,0),2)
                        cv2.imwrite(UPLOAD_FOLDER_lenfosit+'lenfosit_' +time +str(segment_id)+'.jpg', roi)
             	elif result[1]=='eritrosit':
                 	cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0),2)
                        cv2.imwrite(UPLOAD_FOLDER_eritrosit+'eritrosit_' +time+str(segment_id)+'.jpg', roi)
             	elif result[1]=='monosit':
                 	cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0),2)
                        cv2.imwrite(UPLOAD_FOLDER_monosit +'monosit_'+time+str(segment_id)+'.jpg', roi)
             	elif result[1]=='notrofil':
                 	cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,255),2)
                        cv2.imwrite(UPLOAD_FOLDER_notrofil +'notrofil_'+time+str(segment_id)+'.jpg', roi)
             	elif result[1]=='trombosit':
                 	cv2.rectangle(image, (x, y), (x+w, y+h), (203,102,255),2)
                        cv2.imwrite(UPLOAD_FOLDER_trombosit +'trombosit_'+time+str(segment_id)+'.jpg', roi)
	# draw a circle enclosing the object
	     #((x, y), r) = cv2.minEnclosingCircle(c)
	     #cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
	     #cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
             #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                segment_id = segment_id +1
        filename = os.path.join(UPLOAD_FOLDER, 'result.jpg')				
        cv2.imwrite(filename, image)
        return exifutil.open_oriented_im(filename)


    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    #


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((1200, 900))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )

def transform_img(img, img_width=121, img_height=121):

     #Histogram Equalization
     img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
     img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
     img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

     #Image Resizing
     img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

     return img


class ImagenetClassifier(object):
    default_args = {
        'model_def_file': (
            'D:\\web_AI\\AI\\caffenet_deploy_1.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            'D:\\web_AI\\AI\\deep_output.caffemodel'.format(REPO_DIRNAME)),
        'mean_file': (
            'D:\\web_AI\\AI\\mean.binaryproto'.format(REPO_DIRNAME)),
        'class_labels_file': (
            'D:\\web_AI\\AI\\dataset_py.txt'.format(REPO_DIRNAME)),
        'bet_file': (
            'D:\\web_AI\\AI\\imagenet.bet.pickle'.format(REPO_DIRNAME)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 121
    default_args['raw_scale'] = 121.

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        

        mean_blob = caffe_pb2.BlobProto()
        data = open(mean_file, "rb").read()
        mean_blob.ParseFromString(data)

        mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
                  (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
        self.net = caffe.Net(model_def_file,
                pretrained_model_file,
                caffe.TEST)

#Define image transformers
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_mean('data', mean_array)
        self.transformer.set_transpose('data', (2,0,1))

        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort_values('synset_id')['name'].values

        print self.labels




    def classify_image(self,image):
        try:
            test_ids = []
            preds = []
            starttime = time.time()
	    img = transform_img(image)
            self.net.blobs['data'].data[...] = self.transformer.preprocess('data', img)
            out = self.net.forward()
            endtime = time.time()

            pred_probas = out['prob']
	    preds = preds + [pred_probas.argmax()]

            if pred_probas[0][preds[0]] >0.7:
                  predictions = self.labels[preds[0]]
            else:
                  predictions = 'null'
            #print predictions

            logging.info('result: %s', str(predictions))

            return (True, predictions, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    #app.clf.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)

		


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    file_1 = os.getcwd() +'\\'+str('images\\abc_7898.jpg')
    start_from_terminal(app)
