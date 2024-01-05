import numpy as np
import sys, os
cwd_split = os.getcwd().split(os.sep)
root = os.path.abspath("")
sys.path.insert(0, f"{root}/build/python")

import mitsuba as mi
# mi.set_variant("cuda_ad_rgb")
mi.set_variant("cuda_ad_rgb_double")
import drjit as dr

_LOG_DEBUG = True

def JT_J_x(x_eval, x_input, sc):
    """
    Perform forward AD to compute $J x$ and then backward AD to compute $J^T J
    x$. The forward AD estimates the change of y if we perturb x (currently at
    `x_eval`) by `x_input`.
    """

    alpha = mi.Float(0)
    dr.enable_grad(alpha)

    x = x_eval + alpha * x_input
    y = sc.eval_func(x, forward_param=alpha)
    dr.forward_to(y)

    J_x = dr.grad(y)
    
    # Clean up
    dr.disable_grad(alpha)
    if _LOG_DEBUG:
        print(f"J_x:    x_eval = {x_eval}, \n\tx_input = {x_input}, \n\tJ_x = {J_x}")

    return JT_y(x_eval, J_x, sc)


def JT_y(x_eval, y_input, sc):
    """
    Perform backward AD to compute the Jacobian transpose. The backward AD
    estimates that if we want y to change by `y_input`, how should we perturb x
    (currently at `x_eval`).
    """

    x = mi.Float(x_eval)
    dr.enable_grad(x)
    print("x_grad: ", dr.grad_enabled(x))

    y = sc.eval_func(x)
    print("y: ", y)
    print("y_input: ", y_input)
    print("y_grad: ", dr.grad_enabled(y))
    print("y_input_grad: ", dr.grad_enabled(y_input))
    dr.backward(y * y_input)

    res = dr.grad(x)

    # Clean up
    dr.disable_grad(x)
    if _LOG_DEBUG:
        print(f"JT_y:   x_eval = {x_eval}, \n\ty_input = {y_input}, \n\tres = {res}")

    return res


def conjugate_gradient(x_eval, residual, sc):
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
    r_curr = -JT_y(x_eval, residual, sc)    # [xs, ]: the residual of conjugate gradient
    d_curr = mi.Float(r_curr)               # [xs, ]: the conjugate direction

    # ---------- Iterative Conjugate Gradient ----------
    for i in range(10):

        # Intermediate result: J^T J d
        Hd = JT_J_x(x_eval, d_curr, sc)

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
        r_next = r_curr - alpha * Hd

        # DEBUG
        if True:
            print(f"alpha = {alpha}")
            print(f"   | r_curr = {r_curr}")
            print(f"   | r_next = {r_next}")
            print(f"   | dir_x  = {dir_x}")
            print(f"   | d_curr = {d_curr}")

        # Check convergence
        if True:
            # Need a better convergence criterion
            if dr.slice(dr.norm(r_next)) < min(0.5, dr.slice(dr.sqrt(dr.norm(r_curr)))) * dr.slice(dr.norm(r_curr)):
                print(f"Conjugate Gradient converged after {i} iterations")
                print("Current residual is: ", dr.slice(dr.norm(r_next)))
                break

        # Compute the new conjugate direction from the old one weighted by
        # ``beta`` to ensure conjugacy
        beta = dr.squared_norm(r_next) / dr.squared_norm(r_curr)
        d_curr = r_next + beta * d_curr

        # Update
        r_curr = r_next

    # Restore the scene to the original state
    sc.restore()

    return dir_x

class SceneManager():
    """
    A manager that handles the parameter types. But we don't need the grad
    enable feature of common optimizers.
    """
    def __init__(self, scene, params, key):
        # For simplicity, we only use one variable (and one key)
        self.params = params
        self. scene = scene
        #self.key = None
        for k in key:
            self.key = k

    def values(self):
        return dr.ravel(self.params[self.key])

    def update(self, values):
        self.params[self.key] = dr.unravel(type(self.params[self.key]), values)
        self.params.update()

    def save(self):
        self.value_backup = dr.detach(type(dr.ravel(self.params[self.key]))(
                dr.ravel(self.params[self.key])))

    def restore(self):
        self.params[self.key] = dr.unravel(type(self.params[self.key]), self.value_backup)
        self.params.update()

    def step(self, descent_dir):
        curr_val = self.values()
        new_val = curr_val + descent_dir
        self.update(new_val)

    def eval_func(self):
        raise NotImplementedError

class AlbedoScene(SceneManager):
    def __init__(self, scene, params, key, _EXP_IDX):
        super().__init__(scene, params, key)
        self._EXP_IDX = _EXP_IDX

    def eval_func(self, x_data=None, forward_param=None):
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

        # Render the scene
        image = mi.render(self.scene, self.params, spp=512)

        # Loss function: this can be the difference with the reference image, a
        # neural network loss, or any analytical loss function for testing.
        if True:
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

        return loss

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
        curr_val = self.values()
        new_val = curr_val + descent_dir

        # Clamp the value to [0, 1]
        new_val = dr.clamp(new_val, 0.0, 1.0)

        self.update(new_val)
        
        
        