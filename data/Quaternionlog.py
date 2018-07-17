@classmethod

import numpy as np


class Quaternion:
    """Class to represent a 4-dimensional complex number or quaternion.
    Quaternion objects can be used generically as 4D numbers,
    or as unit quaternions to represent rotations in 3D space.
    Attributes:
        q: Quaternion 4-vector represented as a Numpy array
    """

    def __init__(self, *args, **kwargs):
        """Initialise a new Quaternion object.
        See Object Initialisation docs for complete behaviour:

        http://kieranwynn.github.io/pyquaternion/initialisation/

        """
        s = len(args)
        if s is 0:
            # No positional arguments supplied
            if len(kwargs) > 0:
                # Keyword arguments provided
                if ("scalar" in kwargs) or ("vector" in kwargs):
                    scalar = kwargs.get("scalar", 0.0)
                    if scalar is None:
                        scalar = 0.0
                    else:
                        scalar = float(scalar)

                    vector = kwargs.get("vector", [])
                    vector = self._validate_number_sequence(vector, 3)

                    self.q = np.hstack((scalar, vector))
                elif ("real" in kwargs) or ("imaginary" in kwargs):
                    real = kwargs.get("real", 0.0)
                    if real is None:
                        real = 0.0
                    else:
                        real = float(real)

                    imaginary = kwargs.get("imaginary", [])
                    imaginary = self._validate_number_sequence(imaginary, 3)

                    self.q = np.hstack((real, imaginary))
                elif ("axis" in kwargs) or ("radians" in kwargs) or ("degrees" in kwargs) or ("angle" in kwargs):
                    try:
                        axis = self._validate_number_sequence(kwargs["axis"], 3)
                    except KeyError:
                        raise ValueError(
                            "A valid rotation 'axis' parameter must be provided to describe a meaningful rotation."
                        )
                    angle = kwargs.get('radians') or self.to_radians(kwargs.get('degrees')) or kwargs.get(
                        'angle') or 0.0
                    self.q = Quaternion._from_axis_angle(axis, angle).q
                elif "array" in kwargs:
                    self.q = self._validate_number_sequence(kwargs["array"], 4)
                elif "matrix" in kwargs:
                    self.q = Quaternion._from_matrix(kwargs["matrix"]).q
                else:
                    keys = sorted(kwargs.keys())
                    elements = [kwargs[kw] for kw in keys]
                    if len(elements) is 1:
                        r = float(elements[0])
                        self.q = np.array([r, 0.0, 0.0, 0.0])
                    else:
                        self.q = self._validate_number_sequence(elements, 4)

            else:
                # Default initialisation
                self.q = np.array([1.0, 0.0, 0.0, 0.0])
        elif s is 1:
            # Single positional argument supplied
            if isinstance(args[0], Quaternion):
                self.q = args[0].q
                return
            if args[0] is None:
                raise TypeError("Object cannot be initialised from " + str(type(args[0])))
            try:
                r = float(args[0])
                self.q = np.array([r, 0.0, 0.0, 0.0])
                return
            except TypeError:
                pass  # If the single argument is not scalar, it should be a sequence

            self.q = self._validate_number_sequence(args[0], 4)
            return

        else:
            # More than one positional argument supplied
            self.q = self._validate_number_sequence(args, 4)



def log(cls, q):
    """Quaternion Logarithm.
    Find the logarithm of a quaternion amount.
    Params:
         q: the input quaternion/argument as a Quaternion object.
    Returns:
         A quaternion amount representing log(q) := (log(|q|), v/|v|acos(w/|q|)).
    Note:
        The method computes the logarithm of general quaternions. See [Source](https://math.stackexchange.com/questions/2552/the-logarithm-of-quaternion/2554#2554) for more details.
    """
    v_norm = np.linalg.norm(q.vector)
    q_norm = q.norm
    tolerance = 1e-17
    if q_norm < tolerance:
        # 0 quaternion - undefined
        return Quaternion(scalar=-float('inf'), vector=float('nan') * q.vector)
    if v_norm < tolerance:
        # real quaternions - no imaginary part
        return Quaternion(scalar=log(q_norm), vector=[0, 0, 0])
    vec = q.vector / v_norm
    return Quaternion(scalar=log(q_norm), vector=acos(q.scalar / q_norm) * vec)