import jax
import jax.numpy as jnp

# maps 4-dim space to a 3-dim vector sapce 
def f(x): 
    u = jnp.array([
        x[0]**6 * x[1]**4 * x[2]**9 * x[3]**2, 
        x[0]**2 * x[1]**3 * x[2]**5 * x[3]**3, 
        x[0]**5 * x[1]**7 * x[2]**7 * x[3]**6,
    ])
    return u


def main():
  evaluation_point = jnp.array([1.0, 0.5, 1.5, 2.0])
  
  # JAX allows us to easily obtain the Jacobian (a matrix that is given by the 
  # deriavtive of the function w.r.t its input) --> df/dx --> we expect this matrix to be a 3*4-dim matrix
  full_jacobian = jax.jacfwd(f)(evaluation_point)

  # Naive implementation
  # JVP --> using full jacobian and form a matrix vector product with a vector from the right --> df/dx @ v
  multiplication_point = jnp.array([0.2, 0.3, 0.4, 0.8]) # since we are right multiplying it, this one has to be 4-dim

  JVP = full_jacobian @ multiplication_point # 3-dim

  # Using JAX function to obtain the result of the jvp without rxplicitly computing the Jacobian matrix
  
  # jax.jvp(f --> func we want to take the Jacobian, 
  #         primals --> the points in which the Jacobian should be evaluated, 
  #         tangents --> the vector we want to multiply)
  # we have to pass in tuples for primals and tangents since we could potentially have multiple inputs to the function 
  f_evaluated, jvp_evaluated = jax.jvp(f, primals=(evaluation_point, ), tangents=(multiplication_point, ))

if __name__ == "__main__": 
  main()
