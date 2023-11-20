# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""The Projected Gamma distribution class."""

# Dependency imports
import numpy as np
# import tensorflow.compat.v2 as tf
import tensorflow as tf

from tensorflow_probability.python.bijectors import softmax_centered as softmax_centered_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.bijectors import softmax as softmax_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma as gamma_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'ProjectedGamma',
    'FiniteMixProjectedGamma',
]

_dirichlet_sample_note = """Note: `value` must be a non-negative tensor with
dtype `self.dtype` and be in the `(self.event_shape() - 1)`-simplex, i.e.,
`tf.reduce_sum(value, -1) = 1`. It must have a shape compatible with
`self.batch_shape() + self.event_shape()`."""

class FiniteMixProjectedGamma(distribution.AutoCompositeTensorDistribution):
    def __init__(
            self, 
            shape, 
            mixprob, 
            power = 10, 
            validate_args = False, 
            allow_nan_stats = True, 
            force_probs_to_zero_outside_support = False, 
            name = 'FiniteMixProjectedGamma',
        ):
        parameters = dict(locals())
        self._force_probs_to_zero_outside_support = (
            force_probs_to_zero_outside_support
            )
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([shape], dtype_hint = tf.float32)
        self._shape = tensor_util.convert_nonref_to_tensor(shape, dtype = dtype, name = 'shape')
        self._mixprob = tensor_util.convert_nonref_to_tensor(shape, dtype = dtype, name = 'mixprob')
        self._power = power
        super(FiniteMixProjectedGamma, self).__init__(
            dtype = self._shape.dtype,
            validate_args = validate_args,
            allow_nan_stats = allow_nan_stats,
            reparameterization_type = reparameterization.FULLY_REPARAMETERIZED,
            name = name,
            )
        return
    
    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            scale = parameter_properties.ParameterProperties(
                event_ndims=2,
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))
                    )
                ),
            mixprob = parameter_properties.ParameterProperties(
                event_ndims = 1,
                default_contsraining_bijector_fn=(
                    lambda: softmax_bijector.Softmax(low=dtype_util.eps(dtype))
                    )
                )
            )
    
    @property 
    def shape(self):
        return self._shape
    
    @property
    def mixprob(self):
        return self._mixprob

    @property
    def power(self):
        return self._power
    
    @property
    def force_probs_to_zero_outside_support(self):
        return self._force_probs_to_zero_outside_support

    # dunno what's going on here.  check later.
    def _event_shape_tensor(self):
        shape = tf.convert_to_tensor(self.shape)
        return ps.shape(shape)[-1:]
    def _event_shape(self):
        return tensorshape_util.with_rank(self.shape.shape[-1:], rank = 1)

    def _sample_n(self, n, seed = None):
        gamma_sample = gamma_lib.random_gamma()
    
    # @distribution_util.AppendDocstring(_dirichlet_sample_note)
    # def _log_prob(self, x):
    #     shape = self.


class ProjectedGamma(distribution.AutoCompositeTensorDistribution):
    def __init__(
            self,
            concentration,
            p = 10,
            validate_args = False,
            allow_nan_stats = True,
            force_probs_to_zero_outside_support = False,
            name = 'ProjectedGamma',
            ):
        parameters = dict(locals())
        self._force_probs_to_zero_outside_support = (
            force_probs_to_zero_outside_support
            )
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([concentration], dtype_hint=tf.float32)
        self._concentration = tensor_util.convert_nonref_to_tensor(
            concentration, dtype=dtype, name='concentration',
            )
        self._power = p
        super(ProjectedGamma, self).__init__(
            dtype = self._concentration.dtype,
            validate_args = validate_args,
            allow_nan_stats = allow_nan_stats,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            parameters = parameters,
            name = name,
            )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            concentration=parameter_properties.ParameterProperties(
                event_ndims=1,
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))
                    )
                )
            )

    @property
    def concentration(self):
        return self._concentration

    @property
    def power(self):
        return self._power

    @property
    def force_probs_to_zero_outside_support(self):
        return self._force_probs_to_zero_outside_support

    def _event_shape_tensor(self):
        concentration = tf.convert_to_tensor(self.concentration)
        return ps.shape(concentration)[-1:]

    def _event_shape(self):
        return tensorshape_util.with_rank(self.concentration.shape[-1:], rank=1)

    def _sample_n(self, n, seed=None):
        gamma_sample = gamma_lib.random_gamma(
            shape=[n], concentration = self.concentration, 
            seed = seed, log_space = False,
            )
        return tf.math.divide(
            gamma_sample, 
            tf.math.pow(
                tf.math.reduce_sum(
                    tf.math.pow(gamma_sample, self.power), 
                    axis = -1, 
                    keepdims = True,
                    ), 
                1 / self.power,
                )
            )

    @distribution_util.AppendDocstring(_dirichlet_sample_note)
    def _log_prob(self, x):
        concentration = tf.convert_to_tensor(self.concentration)
        lp = tf.reduce_sum(tf.math.xlogy(concentration - 1., x), axis = -1)
        lp -= tf.math.lbeta(concentration)
        lp -= tf.math.xlogy(
            tf.math.reduce_sum(concentration, axis = -1),
            tf.math.reduce_sum(x, axis = -1),
            )
        if self._force_probs_to_zero_outside_support:
            eps = np.finfo(dtype_util.as_numpy_dtype(x.dtype)).eps
            is1 = tf.math.abs(tf.math.reduce_sum(tf.math.pow(x, self.power), axis = -1) - 1. < 1e3 * eps)
            is2 = tf.reduce_all(x >= 0, axis = -1)
            in_support = (is1 & is2)
            return tf.where(in_support, lp, -float('inf'))
        return lp

    @distribution_util.AppendDocstring(_dirichlet_sample_note)
    def _prob(self, x):
        return tf.exp(self._log_prob(x))

    def _default_event_space_bijector(self):
        # TODO(b/145620027) Finalize choice of bijector.
        return softmax_centered_bijector.SoftmaxCentered(
            validate_args=self.validate_args)

    def _sample_control_dependencies(self, x):
        """Checks the validity of a sample."""
        assertions = []
        if not self.validate_args:
            return assertions
        assertions.append(assert_util.assert_non_negative(
            x, 
            message='Samples must be non-negative.'),
            )
        assertions.append(
            assert_util.assert_near(
                tf.ones([], dtype=self.dtype),
                tf.reduce_sum(tf.math.pow(x, self.power), axis = -1),
                message='Sample last dimension must be on S_p^{d-1} Sphere',
                )
            )
        return assertions

    def _parameter_control_dependencies(self, is_init):
        """Checks the validity of the concentration parameter."""
        assertions = []

        # In init, we can always build shape and dtype checks because
        # we assume shape doesn't change for Variable backed args.
        if is_init:
            if not dtype_util.is_floating(self.concentration.dtype):
                raise TypeError('Argument `concentration` must be float type.')

        msg = 'Argument `concentration` must have rank at least 1.'
        ndims = tensorshape_util.rank(self.concentration.shape)
        if ndims is not None:
            if ndims < 1:
                raise ValueError(msg)
            elif self.validate_args:
                assertions.append(
                    assert_util.assert_rank_at_least(
                        self.concentration, 1, message = msg
                        )
                    )
        msg = 'Argument `concentration` must have `event_size` at least 2.'
        event_size = tf.compat.dimension_value(self.concentration.shape[-1])
        if event_size is not None:
            if event_size < 2:
                raise ValueError(msg)
            elif self.validate_args:
                assertions.append(
                    assert_util.assert_less(
                        1, tf.shape(self.concentration)[-1], message = msg,
                        )
                    )
        if not self.validate_args:
            assert not assertions  # Should never happen.
            return []
        if is_init != tensor_util.is_ref(self.concentration):
            assertions.append(
                assert_util.assert_positive(
                    self.concentration,
                    message = 'Argument `concentration` must be positive.',
                    ),
                )
        return assertions

# EOF