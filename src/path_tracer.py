import math
import sys


def siddon_jacobs_3d(p1: tuple, p2: tuple, phan_size: int) -> tuple[list, list]:
  planes = phan_size + 1
  # EDGE CASE: point 1 is 1 or 2 pixels from the edge of the phantom

  d_12: float = 0 # Line integral of the attenuation coefficient
  voxels_to_traverse: list[tuple] = []
  voxels_to_traverse_path_lens: list[float] = []

  # The corner of the pixel space where the axes intersect
  b_x = 0
  b_y = 0
  b_z = 0

  # Distance between planes
  d_x = 1
  d_y = 1
  d_z = 1

  def calc_alpha_x(i: int) -> float:
    numerator = (b_x + i * d_x) - p1[0]
    denominator = p2[0] - p1[0] + sys.float_info.epsilon
    return numerator / denominator

  def calc_alpha_y(j: int) -> float:
    numerator = (b_y + j * d_y) - p1[1]
    denominator = p2[1] - p1[1] + sys.float_info.epsilon
    return numerator / denominator

  def calc_alpha_z(k: int) -> float:
    numerator = (b_z + k * d_z) - p1[2]
    denominator = p2[2] - p1[2] + sys.float_info.epsilon
    return numerator / denominator

  alpha_x_min = min(calc_alpha_x(0), calc_alpha_x(planes-1))
  alpha_x_max = max(calc_alpha_x(0), calc_alpha_x(planes-1))
  alpha_y_min = min(calc_alpha_y(0), calc_alpha_y(planes-1))
  alpha_y_max = max(calc_alpha_y(0), calc_alpha_y(planes-1))
  alpha_z_min = min(calc_alpha_z(0), calc_alpha_z(planes-1))
  alpha_z_max = max(calc_alpha_z(0), calc_alpha_z(planes-1))

  alpha_min = max(0, alpha_x_min, alpha_y_min, alpha_z_min)
  alpha_max = min(1, alpha_x_max, alpha_y_max, alpha_z_max)

  def p_x(alpha: float) -> float:
    return p1[0] + alpha * (p2[0] - p1[0])

  def p_y(alpha: float) -> float:
    return p1[1] + alpha * (p2[1] - p1[1])

  def p_z(alpha: float) -> float:
    return p1[2] + alpha * (p2[2] - p1[2])

  def phi_x(alpha: float) -> float:
    numerator = p_x(alpha) - b_x
    denominator = d_x
    return numerator / denominator

  def phi_y(alpha: float) -> float:
    numerator = p_y(alpha) - b_y
    denominator = d_y
    return numerator / denominator

  def phi_z(alpha: float) -> float:
    numerator = p_z(alpha) - b_z
    denominator = d_z
    return numerator / denominator

  # Determine i_min and i_max, j_min and j_max, k_min and k_max

  i_min: int | None = None
  i_max: int | None = None
  j_min: int | None = None
  j_max: int | None = None
  k_min: int | None = None
  k_max: int | None = None

  if p1[0] < p2[0]:
    i_min = 1 if alpha_min == alpha_x_min else math.ceil(phi_x(alpha_min))
    i_max = planes - 1 if alpha_max == alpha_x_max else math.floor(phi_x(alpha_max))
  else:
    i_max = planes - 2 if alpha_min == alpha_x_min else math.floor(phi_x(alpha_min))
    i_min = 0 if alpha_max == alpha_x_max else math.ceil(phi_x(alpha_max))

  if p1[1] < p2[1]:
    j_min = 1 if alpha_min == alpha_y_min else math.ceil(phi_y(alpha_min))
    j_max = planes - 1 if alpha_max == alpha_y_max else math.floor(phi_y(alpha_max))
  else:
    j_max = planes - 2 if alpha_min == alpha_y_min else math.floor(phi_y(alpha_min))
    j_min = 0 if alpha_max == alpha_y_max else math.ceil(phi_y(alpha_max))

  if p1[2] < p2[2]:
    k_min = 1 if alpha_min == alpha_z_min else math.ceil(phi_z(alpha_min))
    k_max = planes - 1 if alpha_max == alpha_z_max else math.floor(phi_z(alpha_max))
  else:
    k_max = planes - 2 if alpha_min == alpha_z_min else math.floor(phi_z(alpha_min))
    k_min = 0 if alpha_max == alpha_z_max else math.ceil(phi_z(alpha_max))

  # print(i_min, i_max, j_min, j_max, k_min, k_max)

  alpha_x: float | None = None
  alpha_y: float | None = None
  alpha_z: float | None = None

  if p1[0] < p2[0]:
    alpha_x = calc_alpha_x(i_min)
  elif p1[0] == p2[0]:
    alpha_x = alpha_x_max
  else:
    alpha_x = calc_alpha_x(i_max)

  if p1[1] < p2[1]:
    alpha_y = calc_alpha_y(j_min)
  elif p1[1] == p2[1]:
    alpha_y = alpha_y_max
  else:
    alpha_y = calc_alpha_y(j_max)

  if p1[2] < p2[2]:
    alpha_z = calc_alpha_z(k_min)
  elif p1[2] == p2[2]:
    alpha_z = alpha_z_max
  else:
    alpha_z = calc_alpha_z(k_max)
  
  i: int | None = None
  if True:
    numerator = min(alpha_x, alpha_y, alpha_z) + alpha_min
    denominator = 2
    i = math.floor(phi_x(numerator / denominator))
  if True:
    numerator = min(alpha_x, alpha_y, alpha_z) + alpha_min
    denominator = 2
    j = math.floor(phi_y(numerator / denominator))
  if True:
    numerator = min(alpha_x, alpha_y, alpha_z) + alpha_min
    denominator = 2
    k = math.floor(phi_z(numerator / denominator))

  # Update alpha_x, alpha_y, alpha_z and i j k
  # Accoding to whether we cross an x, y or z plane

  a_xu = d_x / (abs(p2[0] - p1[0]) + sys.float_info.epsilon)
  a_yu = d_y / (abs(p2[1] - p1[1]) + sys.float_info.epsilon)
  a_zu = d_z / (abs(p2[2] - p1[2]) + sys.float_info.epsilon)

  i_u: int = 1 if p1[0] < p2[0] else -1
  j_u: int = 1 if p1[1] < p2[1] else -1
  k_u: int = 1 if p1[2] < p2[2] else -1

  alpha_c = alpha_min
  d_conv = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

  while alpha_c < alpha_max:

    i_to_add: int = i
    j_to_add: int = j
    k_to_add: int = k

    length: float = 0

    if (alpha_x <= alpha_y) and (alpha_x <= alpha_z):
      length = (alpha_x - alpha_c) * d_conv

      alpha_c = alpha_x
      alpha_x = alpha_x + a_xu
      
      i = i + i_u

    elif (alpha_y <= alpha_x) and (alpha_y <= alpha_z):
      length = (alpha_y - alpha_c) * d_conv

      alpha_c = alpha_y
      alpha_y = alpha_y + a_yu

      j = j + j_u
      
    else: # alpha_z is smallest
      length = (alpha_z - alpha_c) * d_conv

      alpha_c = alpha_z
      alpha_z = alpha_z + a_zu
      k = k + k_u

    if alpha_c > alpha_max:
      break
    
    if length > 0:
      voxels_to_traverse.append((i_to_add, j_to_add, k_to_add))
      voxels_to_traverse_path_lens.append(length)

  return voxels_to_traverse, voxels_to_traverse_path_lens
