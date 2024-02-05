import numpy as np
import sys, os
cwd_split = os.getcwd().split(os.sep)
root = os.path.abspath("")
sys.path.insert(0, f"{root}/build/python")

import mitsuba as mi
# mi.set_variant("cuda_ad_rgb")
mi.set_variant("cuda_ad_rgb_double")
import drjit as dr

_LOG_DEBUG = False

def JT_J_x(x_eval, x_input, sc):
    """
    Perform forward AD to compute $J x$ and then backward AD to compute $J^T J
    x$. The forward AD estimates the change of y if we perturb x (currently at
    `x_eval`) by `x_input`.
    """

    alpha = mi.Float(0)
    dr.enable_grad(alpha)

    x = x_eval + alpha * x_input
    image, y = sc.eval_func(x, forward_param=alpha)
    dr.forward_to(y)

    J_x = dr.grad(y)
    
    # Clean up
    dr.disable_grad(alpha)
    if _LOG_DEBUG:
        print(f"J_x:    x_eval = {x_eval}, \n\tx_input = {x_input}, \n\tJ_x = {J_x}")

    Hd = JT_y(x_eval, J_x, sc)
    return Hd


def JT_y(x_eval, y_input, sc):
    """
    Perform backward AD to compute the Jacobian transpose. The backward AD
    estimates that if we want y to change by `y_input`, how should we perturb x
    (currently at `x_eval`).
    """

    x = mi.Float(x_eval)
    dr.enable_grad(x)
    # print("x_grad: ", dr.grad_enabled(x))

    image, y = sc.eval_func(x)
    # print("y: ", y)
    # print("y_input: ", y_input)
    # print("y_grad: ", dr.grad_enabled(y))
    # print("y_input_grad: ", dr.grad_enabled(y_input))
    dr.backward(y * y_input)

    res = dr.grad(x)

    # Clean up
    dr.disable_grad(x)
    if _LOG_DEBUG:
        print(f"JT_y:   x_eval = {x_eval}, \n\ty_input = {y_input}, \n\tres = {res}")

    return res


def conjugate_gradient_line(x_eval, residual, sc):
    """
    Computes Gauss-Newton descent direction `(J^T J)^{-1} J^T residual` using
    Conjugate Gradient. 

    Parameter ``x_eval`` [xs, ]:
        The current value of x.
    
    Parameter ``residual``:
        The current value of the residual evaluated with ``func(x_eval)``.

    Output ``dir_x`` [xs, ]:
        The descent direction.
    """

    # Save the current scene state
    sc.save()

    # ---------- Intialize ----------
    x_eval = mi.Float(dr.detach(x_eval))
    xs = dr.width(x_eval)
    dir_x = dr.zeros(mi.Float, xs)          # [xs, ]: the descent direction to be computed
    # FIXME
    r_curr = JT_y(x_eval, residual, sc)    # [xs, ]: the residual of conjugate gradient
    sc.set_gradient(r_curr)
    d_curr = -mi.Float(r_curr)               # [xs, ]: the conjugate direction

    # ---------- Iterative Conjugate Gradient ----------
    for i in range(10):

        # Intermediate result: J^T J d
        Hd = JT_J_x(x_eval, d_curr, sc)
        dHd = dr.dot(d_curr, Hd)
        if dHd[0] <= 0:
            #if _LOG_DEBUG:
            print("dHd: ", dHd)
            print("d: ", d_curr)
            print("Hd: ", Hd)
            if i == 0:
                return -r_curr
            else:
                return dir_x
        
        # Compute the step size of how far the descent direction should move
        # along the conjugate direction: alpha = (r^T r) / (d^T J^T J d).
        # (1) The denominator is d^T J^T J d -- measuring how much the conjugate
        # direction aligns with the curvature (Hessian) approximated by J^T J.
        # (2) The numerator is r^T r -- not the typical r^T d in conjugate
        # gradient, because we are solving a Gauss-Newton problem. For example
        # see "Dembo, R.S., Steihaug, T. Truncated-Newton algorithms for
        # large-scale unconstrained optimization."
        alpha = dr.squared_norm(r_curr) / \
                dr.dot(d_curr, Hd)

        # Update the descent direction using the conjugate direction
        dir_x += alpha * d_curr

        # Compute the new conjugate gradient residual: the estimated change in
        # the residual if we move from the current point in the parameter space
        # along the direction d_curr by a step size of alpha.
        r_next = r_curr + alpha * Hd

        # DEBUG
        if _LOG_DEBUG:
            print(f"alpha = {alpha}")
            print(f"   | r_curr = {r_curr}")
            print(f"   | r_next = {r_next}")
            print(f"   | dir_x  = {dir_x}")
            print(f"   | d_curr = {d_curr}")

        # Check convergence
        if True:
            # Need a better convergence criterion
            if dr.slice(dr.norm(r_next)) < min(0.5, dr.slice(dr.sqrt(dr.norm(r_curr)))) * dr.slice(dr.norm(r_curr)):
                if _LOG_DEBUG:
                    print(f"Conjugate Gradient converged after {i} iterations")
                    print("Current residual is: ", dr.slice(dr.norm(r_next)))
                break

        # Compute the new conjugate direction from the old one weighted by
        # ``beta`` to ensure conjugacy
        beta = dr.squared_norm(r_next) / dr.squared_norm(r_curr)
        d_curr = -r_next + beta * d_curr

        # Update
        r_curr = r_next
        sc.set_gradient(r_curr)

    # Restore the scene to the original state
    sc.restore()

    return dir_x

class SceneManager():
    """
    A manager that handles the parameter types. But we don't need the grad
    enable feature of common optimizers.
    """
    def __init__(self, scene, params, keys):
        # For simplicity, we only use one variable (and one key)
        self.params = params
        self.scene = scene
        self.keys = keys
        self.value_backup = {}
        
        #self.value_dict = {}
        self.len_values = {}
        for key in self.keys:
            param_flat = dr.ravel(self.params[key])
            #self.value_dict[key] = param_flat
            self.len_values[key] = len(param_flat)
        self.total_length = dr.sum(self.len_values.values())
        #print(self.total_length)
        #print(self.len_values)
           
        start_i = 0
        self.values_flatten = dr.zeros(mi.Float, self.total_length)
        for key in self.keys:
            param_flat = dr.ravel(self.params[key])
            index = dr.arange(mi.UInt32, start_i, start_i + self.len_values[key])
            dr.scatter(self.values_flatten, param_flat, index)
            #self.values_flatten[start_i: start_i + self.len_values[key]] = self.value_dict[key]
            start_i += self.len_values[key]
    
    def update_init(self, params):
        for key in self.keys:
            param_flat = dr.ravel(params[key])
            #self.value_dict[key] = param_flat
            self.len_values[key] = len(param_flat)
        self.total_length = dr.sum(self.len_values.values())
        #print(self.total_length)
        #print(self.len_values)
           
        start_i = 0
        self.values_flatten = dr.zeros(mi.Float, self.total_length)
        for key in self.keys:
            param_flat = dr.ravel(self.params[key])
            index = dr.arange(mi.UInt32, start_i, start_i + self.len_values[key])
            dr.scatter(self.values_flatten, param_flat, index)
            #self.values_flatten[start_i: start_i + self.len_values[key]] = self.value_dict[key]
            start_i += self.len_values[key]
    
    def values(self):
        # print("values_flatten: ", self.values_flatten)
        # for key in self.keys:
        #     print("param_values: ", dr.ravel(self.params[key]))
        
        # for key in self.keys:
        #     return dr.ravel(self.params[key])
        start_i = 0
        for key in self.keys:
            param_flat = dr.ravel(self.params[key])
            index = dr.arange(mi.UInt32, start_i, start_i + self.len_values[key])
            #print("index: ", index)
            #print("params: ", param_flat)
            #print("values_faltten: ", self.values_flatten)
            dr.scatter(self.values_flatten, param_flat, index)
            #self.values_flatten[start_i: start_i + self.len_values[key]] = self.value_dict[key]
            start_i += self.len_values[key]
            
        return self.values_flatten
    
        

    def update(self, values):
        start_i = 0
        for key in self.keys:
            index = dr.arange(mi.UInt, start_i, start_i + self.len_values[key])
            param_data = dr.gather(mi.Float, values, index)
            param_reshape = dr.unravel(type(self.params[key]), param_data)
            # param_reshape = mi.TensorXf(param_data, shape = self.params[key].shape)
            
            self.params[key] = param_reshape
            start_i += self.len_values[key]
            
            #self.params[key] = dr.unravel(type(self.params[key]), values)
        self.params.update()

    def save(self):
        for key in self.keys:
            self.value_backup[key] = dr.detach(type(dr.ravel(self.params[key]))(
                dr.ravel(self.params[key])))
        # for key in self.keys:
        #     self.value_backup = dr.detach(type(dr.ravel(self.params[key]))(
        #             dr.ravel(self.params[key])))

    def restore(self):
        for key in self.keys:
            data_restore = dr.unravel(type(self.params[key]), self.value_backup[key]) 
            # data_restore = mi.TensorXf(self.value_backup[key], self.params[key].shape)
            self.params[key] = data_restore
            #self.params[key] = dr.unravel(type(self.params[key]), self.value_backup[key])
        self.params.update()
        
        # for key in self.keys:
        #     self.params[key] = dr.unravel(type(self.params[key]), self.value_backup)
        #     self.params.update()

    def step(self, descent_dir):
        curr_val = self.values()
        new_val = curr_val + descent_dir
        self.update(new_val)

    def eval_func(self):
        raise NotImplementedError

class AlbedoScene_line(SceneManager):
    def __init__(self, scene, params, keys, sensor = None, ref_images = None, _EXP_IDX = 0):
        super().__init__(scene, params, keys)
        self._EXP_IDX = _EXP_IDX
        self.ref_images = ref_images
        self.sensor = sensor
        self.gradient = None

    def set_gradient(self, gradient_new):
        self.gradient = gradient_new

    def eval_func(self, x_data=None, forward_param=None, sensor_idx = None):
        """
        Evaluate the function "y=func(x)" that we want to optimize using
        Gauss-Newton. In theory, this should be the same function in both forward
        and backward process. Here we distinguish them because DrJIT requires the
        forward process to propogate the latent variable derivatives to scene
        parameters first.
        """

        if x_data is not None:
            # Update the scene parameters using the provided data
            self.update(x_data)

            # Take care of the AD scope for the forward pass
            if forward_param is not None:
                dr.forward_from(forward_param, flags=dr.ADFlag.ClearEdges)
        else:
            # Render the scene using the current parameter values. No
            # derivatives attached. We use this to compute the residual.
            pass
        if self.sensor != None:
            if sensor_idx != None:
                image = mi.render(self.scene, self.params, sensor=self.sensor[sensor_idx], spp=512)
            else:
                image = mi.render(self.scene, self.params, sensor=self.sensor, spp=512)
            
        else:
            image = mi.render(self.scene, self.params, spp=512)

        # Loss function: this can be the difference with the reference image, a
        # neural network loss, or any analytical loss function for testing.
        if self._EXP_IDX == -1:
            if sensor_idx != None:
                loss = dr.mean(dr.abs(image - self.ref_images[sensor_idx]))
            else:
                loss = dr.mean(dr.abs(image - self.ref_images))
            # print("ref_images_shape: ", self.ref_images[sensor_idx])
            # print("image shape: ", image)
                #raise ValueError('sensor index should be given.')
            # loss *= 1e5  # CRAZY
        elif self._EXP_IDX == -2:
            if sensor_idx != None:
                loss = dr.sum(dr.sqr(image - self.ref_images[sensor_idx]))
            else:
                loss = dr.sum(dr.sqr(image - self.ref_images))
        else:
            r = dr.mean(image[:, :, 0])
            g = dr.mean(image[:, :, 1])
            b = dr.mean(image[:, :, 2])

            if self._EXP_IDX == 0:
                # The following two are equivalent: (1) a vector of squares, or
                # (2) a scalar summation of squares

                loss = mi.Vector3f(dr.sqr(r), dr.sqr(g), dr.sqr(b))
                # loss = dr.sqr(r) + dr.sqr(g) + dr.sqr(b)

            elif self._EXP_IDX == 1:
                loss = mi.Vector3f(dr.sqr(r), dr.sqr(g - 0.2), dr.sqr(b - 1))

            elif self._EXP_IDX == 2:
                # loss = mi.Vector3f(dr.sqr(r + g - 0.8), dr.sqr(g - 0.5), dr.sqr(b - 0.8))
                loss = dr.sqr(r + g - 0.8) + dr.sqr(g - 0.5) + dr.sqr(b - 0.8)

            elif self._EXP_IDX == 3:
                loss = r * g * g + r * r + g - 2

        return image, loss

    def expected_value(self):
        if self._EXP_IDX == 0:
            return f"Minimize: x^2 + y^2 + z^2 \n  Expected: [0, 0, 0]"
        elif self._EXP_IDX == 1:
            return f"Minimize: x^2 + (y - 0.2)^2 + (z - 1)^2 \n  Expected: [0, 0.2, 1]"
        elif self._EXP_IDX == 2:
            return f"Minimize: (x + y - 0.8)^2 + (y - 0.5)^2 + (z - 0.8)^2 \n  Expected: [0.3, 0.5, 0.8]"
        elif self._EXP_IDX == 3:
            return f"Minimize: x * y^2 + x^2 + y - 2 \n  Expected: [?]"

    def step(self, descent_dir):
        prev_val = self.values()
        prev_image, prev_loss = self.eval_func()
        # step_size = 1e6
        # step = 0
        # # f(x + alpha * d) <= f(x) + c1 * alpha * d * gradient
        # while step < 20:
        #     step_size /= 2
        #     val_test = curr_val + descent_dir * step_size
        
        
        step_size=32.0
        max_l_iter=10
        c1=1e-7
        new_val = prev_val
        
        for l_iter in range(max_l_iter):
            print("l_iter: ", l_iter)
            step_size *= 0.5
            new_val = prev_val + step_size * descent_dir
            new_image, new_loss = self.eval_func(new_val)
            armijo = dr.all(new_loss <= prev_loss + c1 * step_size * dr.sum(dr.dot(descent_dir, -self.gradient)))
            print(armijo)
            if armijo or l_iter == max_l_iter-1:
                break

        # new_val = prev_val + descent_dir
        # Clamp the value to [0, 1]
        new_val = dr.clamp(new_val[0:3], 0.0, 1.0)
        # new_val = dr.clamp(new_val, 0.0, 1.0)

        self.update(new_val)
        
        
        