#!/opt/anaconda3/envs/default2/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 12:29:51 2019
ref : https://becominghuman.ai/logging-in-tensorboard-with-pytorch-or-any-other-library-c549163dee9e
@author: ankit
"""

import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.misc
import torch
import math
import matplotlib.pyplot as plt
import io
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def close(self):
        self.writer.close()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def layer_summary(self, tag, value, step):

        tag_ = tag+'/min'
        val = torch.min(value).item()
        self.scalar_summary(tag_, val, step + 1)

        tag_ = tag+'/max'
        val = torch.max(value).item()
        self.scalar_summary(tag_, val, step + 1)

        tag_ = tag+'/mean'
        val = torch.mean(value).item()
        self.scalar_summary(tag_, val, step + 1)

        tag_ = tag+'/std'
        val = torch.std(value).item()
        self.scalar_summary(tag_, val, step + 1)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(
                tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_customplot(self, tag, figure, global_step):
        plot_buf = io.BytesIO()
        figure.savefig(plot_buf, format='png')
        plot_buf.seek(0)
        img = Image.open(plot_buf)
        img_ar = np.array(img)

        img_summary = tf.Summary.Image(
            encoded_image_string=plot_buf.getvalue(),
            height=img_ar.shape[0],
            width=img_ar.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_customimage(self, tag, img, global_step):
        s = io.BytesIO()
        Image.fromarray(img).save(s, format='png')

        img_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_layeroutput(self, tag, value):

        fig = plt.figure()
        ix = 1
        sq = math.ceil(math.sqrt(value.shape[0]))
        for _ in range(sq):
            for _ in range(sq):
                ax = plt.subplot(sq, sq, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(value[ix - 1, :, :], cmap='YlOrRd')
                ix += 1
                if (ix > value.shape[0]):
                    break
            else:
                # Continue if the inner loop wasn't broken.
                continue
            # Inner loop was broken, break the outer.
            break
        plot_buf = io.BytesIO()
        fig.savefig(plot_buf, format='png')
        plot_buf.seek(0)
        img = Image.open(plot_buf)
        img_ar = np.array(img)

        img_summary = tf.Summary.Image(
            encoded_image_string=plot_buf.getvalue(),
            height=img_ar.shape[0],
            width=img_ar.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary)
        self.writer.flush()
        plt.close(fig)
