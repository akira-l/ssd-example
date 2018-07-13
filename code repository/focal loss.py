    def focal_loss(self, labels, logits, gamma=2.0, alpha=4.0):
        """
        focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: logits is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        
        :param labels: ground truth labels, shape of [batch_size]
        :param logits: model's output, shape of [batch_size, num_cls]
        :param gamma:
        :param alpha:
        :return: shape of [batch_size]
        """
        print("labels---------", labels)
        print("logits---------", logits)
        epsilon = 1.e-9
        labels = tf.to_int64(labels)
        labels = tf.convert_to_tensor(labels, tf.int64)
        logits = tf.convert_to_tensor(tf.cast(logits, tf.float32), tf.float32)
        num_cls = logits.shape[1]
        print("========================== logits",logits)
        print("========================== labels",labels)
        #raise Exception("whatever")
        
        model_out = tf.add(logits, epsilon)
        print("========================== model_out",model_out)
        
        onehot_labels = tf.one_hot(labels, num_cls)
        print("========================== onehot_labels",onehot_labels)
        
        raise Exception("fuck")
        ce = tf.multiply(onehot_labels, -tf.log(model_out))
        weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max
        return reduced_fl







    def focal_loss(self,prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
        """Compute focal loss for predictions.
            Multi-labels Focal loss formula:
                FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                     ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
        Args:
         prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
         target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
         weights: A float tensor of shape [batch_size, num_anchors]
         alpha: A scalar tensor for focal loss alpha hyper-parameter
         gamma: A scalar tensor for focal loss gamma hyper-parameter
        Returns:
            loss: A (scalar) tensor representing the value of the loss function
        """
        
        #prediction_tensor = tf.to_float32(prediction_tensor)
        prediction_tensor = tf.convert_to_tensor(tf.cast(prediction_tensor, tf.float32), tf.float32)
        target_tensor = tf.convert_to_tensor(tf.cast(target_tensor, tf.float32), tf.float32)
        print("=============prediction_tensor", prediction_tensor)
        
        print("=============target_tensor",target_tensor)
        
        
        #sigmoid_p = tf.nn.sigmoid(prediction_tensor)
        softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.feature_class, labels=self.groundtruth_class)
        zeros = array_ops.zeros_like(target_tensor, dtype=target_tensor.dtype)
        print("==============zeros", zeros)
        #print("==============sigmoid_p", sigmoid_p)
        print("=================softmax_cross_entropy", softmax_cross_entropy)
        #zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
        #raise Exception("fuck all")
        # For poitive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - softmax_cross_entropy, zeros)
        
        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = array_ops.where(target_tensor > zeros, zeros, softmax_cross_entropy)
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(softmax_cross_entropy, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - softmax_cross_entropy, 1e-8, 1.0))
        return tf.reduce_sum(per_entry_cross_ent)












