def ssd_losses(logits, #预测类别
               localisations,#预测位置
               gclasses, #ground truth 类别
               glocalisations, #ground truth 位置
               gscores,#ground truth 分数
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
    """Loss functions for training the SSD 300 VGG network.

    This function defines the different loss components of the SSD, and
    adds them to the TF loss collection.

    Arguments:
      logits: (list of) predictions logits Tensors;
      localisations: (list of) localisations Tensors;
      gclasses: (list of) groundtruth labels Tensors;
      glocalisations: (list of) groundtruth localisations Tensors;
      gscores: (list of) groundtruth score Tensors;
    """
    # Some debugging...
    # for i in range(len(gclasses)):
    #     print(localisations[i].get_shape())
    #     print(logits[i].get_shape())
    #     print(gclasses[i].get_shape())
    #     print(glocalisations[i].get_shape())
    #     print()
    with tf.name_scope(scope):
        l_cross = []
        l_loc = []
        for i in range(len(logits)):
            with tf.name_scope('block_%i' % i):
                # Determine weights Tensor.
                pmask = tf.cast(gclasses[i] > 0, logits[i].dtype)
                n_positives = tf.reduce_sum(pmask)#正样本数目
                
                #np.prod函数Return the product of array elements over a given axis
                n_entries = np.prod(gclasses[i].get_shape().as_list())
                # r_positive = n_positives / n_entries
                # Select some random negative entries.
                r_negative = negative_ratio * n_positives / (n_entries - n_positives)#负样本数
                nmask = tf.random_uniform(gclasses[i].get_shape(),
                                          dtype=logits[i].dtype)
                nmask = nmask * (1. - pmask)
                nmask = tf.cast(nmask > 1. - r_negative, logits[i].dtype)

                #cross_entropy loss
                # Add cross-entropy loss.
                with tf.name_scope('cross_entropy'):
                    # Weights Tensor: positive mask + random negative.
                    weights = pmask + nmask
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits[i],
                                                                          gclasses[i])
                    loss = tf.contrib.losses.compute_weighted_loss(loss, weights)
                    l_cross.append(loss)

                #smooth loss
                # Add localization loss: smooth L1, L2, ...
                with tf.name_scope('localization'):
                    # Weights Tensor: positive mask + random negative.
                    weights = alpha * pmask
                    loss = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
                    loss = tf.contrib.losses.compute_weighted_loss(loss, weights)
                    l_loc.append(loss)

        # Total losses in summaries...
        with tf.name_scope('total'):
            tf.summary.scalar('cross_entropy', tf.add_n(l_cross))
            tf.summary.scalar('localization', tf.add_n(l_loc))
