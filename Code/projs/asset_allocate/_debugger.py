# import numpy as np

# train_rtn_mat =  np.array(
# [
#     [
#       -0.002199999988079071,
#       0.006200000178068876,
#       0.013100000098347664,
#       0.00039999998989515007,
#       -0.006899999920278788,
#       -0.005400000140070915,
#       -0.00279999990016222,
#       0.002199999988079071,
#       0.005499999970197678,
#       -0.007799999788403511,
#       0.012600000016391277,
#       0.00039999998989515007,
#       -0.008700000122189522,
#       -0.003700000001117587,
#       0.0003000000142492354,
#       -0.009200000204145908,
#       -0.0006000000284984708,
#       -0.0010999999940395355,
#       0.021299999207258224,
#       -0.0026000000070780516,
#       -0.0020000000949949026,
#       0.018400000408291817,
#       0.004600000102072954,
#       0.0,
#       -0.008899999782443047,
#       0.005799999926239252,
#       0.002300000051036477,
#       -0.005900000222027302,
#       -0.0024999999441206455,
#       -0.004900000058114529
#     ],
#     [
#       -0.002199999988079071,
#       0.006200000178068876,
#       0.013100000098347664,
#       0.00039999998989515007,
#       -0.006899999920278788,
#       -0.005400000140070915,
#       -0.00279999990016222,
#       0.002199999988079071,
#       0.005499999970197678,
#       -0.007799999788403511,
#       0.012600000016391277,
#       0.00039999998989515007,
#       -0.008700000122189522,
#       -0.003700000001117587,
#       0.0003000000142492354,
#       -0.009200000204145908,
#       -0.0006000000284984708,
#       -0.0010999999940395355,
#       0.021299999207258224,
#       -0.0026000000070780516,
#       -0.0020000000949949026,
#       0.018400000408291817,
#       0.004600000102072954,
#       0.0,
#       -0.008899999782443047,
#       0.005799999926239252,
#       0.002300000051036477,
#       -0.005900000222027302,
#       -0.0024999999441206455,
#       -0.004900000058114529
#     ],
#     [
#       -0.00019999999494757503,
#       0.010499999858438969,
#       0.007300000172108412,
#       0.0003000000142492354,
#       -0.003800000064074993,
#       -0.007400000002235174,
#       -0.0032999999821186066,
#       0.0017999999690800905,
#       0.006500000134110451,
#       -0.009200000204145908,
#       0.008899999782443047,
#       -0.0032999999821186066,
#       -0.0031999999191612005,
#       0.000699999975040555,
#       0.0007999999797903001,
#       -0.008299999870359898,
#       -0.0008999999845400453,
#       -0.0010999999940395355,
#       0.01720000058412552,
#       -0.0027000000700354576,
#       -0.005900000222027302,
#       0.013100000098347664,
#       0.006800000090152025,
#       -0.0003000000142492354,
#       -0.003100000089034438,
#       0.004900000058114529,
#       0.003000000026077032,
#       -0.006500000134110451,
#       -0.002199999988079071,
#       -0.0052999998442828655
#     ],
#     [
#       -0.00559999980032444,
#       0.0035000001080334187,
#       0.01590000092983246,
#       0.0,
#       -0.007300000172108412,
#       -0.006899999920278788,
#       -0.0031999999191612005,
#       0.00419999985024333,
#       0.005499999970197678,
#       -0.005799999926239252,
#       0.013899999670684338,
#       0.0010999999940395355,
#       -0.009399999864399433,
#       -0.0034000000450760126,
#       0.0006000000284984708,
#       -0.006000000052154064,
#       -0.00019999999494757503,
#       -0.00279999990016222,
#       0.028200000524520874,
#       -0.000699999975040555,
#       -0.0010999999940395355,
#       0.02539999969303608,
#       0.004000000189989805,
#       -0.003100000089034438,
#       -0.008700000122189522,
#       0.009399999864399433,
#       0.002300000051036477,
#       -0.006399999838322401,
#       -0.002300000051036477,
#       -0.003100000089034438
#     ],
#     [
#       -0.007899999618530273,
#       0.0005000000237487257,
#       0.017000000923871994,
#       0.0012000000569969416,
#       -0.006899999920278788,
#       -0.007499999832361937,
#       -0.004699999932199717,
#       0.005499999970197678,
#       0.006099999882280827,
#       -0.0020000000949949026,
#       0.013100000098347664,
#       0.0006000000284984708,
#       -0.010700000450015068,
#       -0.003599999938160181,
#       0.002199999988079071,
#       -0.004100000020116568,
#       0.0010000000474974513,
#       -0.003700000001117587,
#       0.031300000846385956,
#       -0.0007999999797903001,
#       -0.00039999998989515007,
#       0.027799999341368675,
#       0.002400000113993883,
#       -0.00279999990016222,
#       -0.009499999694526196,
#       0.00930000003427267,
#       0.00139999995008111,
#       -0.005400000140070915,
#       -0.0012000000569969416,
#       -0.0010000000474974513
#     ]
#   ]
# )


# for row_num, col_num in zip(*inds):
#     print(row_num)
#     print(col_num)


# constraints = [None, None]
# low_constraints, high_constraints = constraints

# no_bounds = low_constraints is None and high_constraints is None


# # 已经足够画出mean-var曲线

# cov_mat_inv = np.linalg.inv(expct_cov_mat)
# ones = np.ones_like(expct_rtn_rates)

# print(f"expct_cov_mat\n{expct_cov_mat}")
# print(f"cov_mat_inv\n{cov_mat_inv}")

# quad_term = ones @ cov_mat_inv @ ones

# print(f"quad_term {quad_term}")


# const_term = expct_rtn_rates @ cov_mat_inv @ expct_rtn_rates
# print(f"const_term {const_term}")


# lin_term = ones @ cov_mat_inv @ expct_rtn_rates
# lin_term2 = expct_rtn_rates @cov_mat_inv @ ones

# print(f"lin_term {lin_term}, lin_term2 {lin_term2}")



# norm_term = const_term*quad_term - np.power(lin_term, 2)

# # var最小的return-var点是(var=1/c, r=a/c)
# vertex = (1.0/quad_term, lin_term/quad_term)


# # build_quad_program()

# expt_tgt_value = 9.999999747378752e-05

# goal_var = expt_tgt_value
# goal_r = lin_term/quad_term + np.sqrt(norm_term/quad_term *\
#                                       (goal_var + np.power(lin_term, 2)/(norm_term*quad_term) - const_term/norm_term))


# if np.isnan(goal_r):
#     print(f'NaN calculation on __cal_portf_rtn_unbounds_from_var '\
#     f'with goal_var {goal_var}, norm_term {norm_term}, quad_term {quad_term}, '\
#     f'lin_term {lin_term}, const_term {const_term}')

# # print(f'goal_r {goal_r}')

# # ones = np.ones_like(expct_rtn_rates)
# # portf_w = goal_r * 1.0 / norm_term * cov_mat_inv @ ( quad_term * expct_rtn_rates - lin_term * ones ) \
# #         + \
# #         1.0 / norm_term * cov_mat_inv @ ( const_term * ones - lin_term * expct_rtn_rates )

# [[-0.0022  0.0062  0.0131  0.0004 -0.0069 -0.0054 -0.0028  0.0022  0.0055
#   -0.0078  0.0126  0.0004 -0.0087 -0.0037  0.0003 -0.0092 -0.0006 -0.0011
#    0.0213 -0.0026 -0.002   0.0184  0.0046  0.     -0.0089  0.0058  0.0023
#   -0.0059 -0.0025 -0.0049]
#  [-0.0022  0.0062  0.0131  0.0004 -0.0069 -0.0054 -0.0028  0.0022  0.0055
#   -0.0078  0.0126  0.0004 -0.0087 -0.0037  0.0003 -0.0092 -0.0006 -0.0011
#    0.0213 -0.0026 -0.002   0.0184  0.0046  0.     -0.0089  0.0058  0.0023
#   -0.0059 -0.0025 -0.0049]
#  [-0.0002  0.0105  0.0073  0.0003 -0.0038 -0.0074 -0.0033  0.0018  0.0065
#   -0.0092  0.0089 -0.0033 -0.0032  0.0007  0.0008 -0.0083 -0.0009 -0.0011
#    0.0172 -0.0027 -0.0059  0.0131  0.0068 -0.0003 -0.0031  0.0049  0.003
#   -0.0065 -0.0022 -0.0053]
#  [-0.0056  0.0035  0.0159  0.     -0.0073 -0.0069 -0.0032  0.0042  0.0055
#   -0.0058  0.0139  0.0011 -0.0094 -0.0034  0.0006 -0.006  -0.0002 -0.0028
#    0.0282 -0.0007 -0.0011  0.0254  0.004  -0.0031 -0.0087  0.0094  0.0023
#   -0.0064 -0.0023 -0.0031]
#  [-0.0079  0.0005  0.017   0.0012 -0.0069 -0.0075 -0.0047  0.0055  0.0061
#   -0.002   0.0131  0.0006 -0.0107 -0.0036  0.0022 -0.0041  0.001  -0.0037
#    0.0313 -0.0008 -0.0004  0.0278  0.0024 -0.0028 -0.0095  0.0093  0.0014
#   -0.0054 -0.0012 -0.001 ]]

result = \
{
  "assets_id": [
    "000001.SH",
    "000002.SH",
    "000009.SH",
    "000010.SH",
    "000016.SH"
  ],
  "backtest": {
    "annual_rtn": -0.1645698653333022,
    "gross_rtn": -0.13013148307800293,
    "rtn": -0.13013148307800293,
    "std": 0.007475195452570915,
    "total_cost": 0,
    "trade_days": 191,
    "var": 5.5878546845633537e-05
  },
  "benchmark": {
    "annual_rtn": -0.17428960101296398,
    "gross_rtn": -0.13798856735229492,
    "rtn": -0.13798856735229492,
    "std": 0.008076509460806847,
    "total_cost": 0,
    "trade_days": 191,
    "var": 6.523000774905086e-05
  },
  "details": [
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 1,
      "solve_status": "FAIL_Co-Linearity found with assets [['000001.SH' '000002.SH']]"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 2,
      "solve_status": "FAIL_Co-Linearity found with assets [['000001.SH' '000002.SH']]"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 3,
      "solve_status": "FAIL_Co-Linearity found with assets [['000001.SH' '000002.SH']]"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 4,
      "solve_status": "FAIL_Co-Linearity found with assets [['000001.SH' '000002.SH']]"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 5,
      "solve_status": "FAIL_Co-Linearity found with assets [['000001.SH' '000002.SH']]"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 6,
      "solve_status": "FAIL_Co-Linearity found with assets [['000001.SH' '000002.SH']]"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 7,
      "solve_status": "FAIL_Co-Linearity found with assets [['000001.SH' '000002.SH']]"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 8,
      "solve_status": "FAIL_minimum expected target variance value(after dilate) for this process is 0.357. Raise goal variance"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 9,
      "solve_status": "FAIL_minimum expected target variance value(after dilate) for this process is 0.371. Raise goal variance"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 10,
      "solve_status": "FAIL_minimum expected target variance value(after dilate) for this process is 0.373. Raise goal variance"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 11,
      "solve_status": "FAIL_Co-Linearity found with assets [['000001.SH' '000002.SH']]"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 12,
      "solve_status": "FAIL_Co-Linearity found with assets [['000001.SH' '000002.SH']]"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 13,
      "solve_status": "FAIL_Co-Linearity found with assets [['000001.SH' '000002.SH']]"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 14,
      "solve_status": "FAIL_minimum expected target variance value(after dilate) for this process is 0.586. Raise goal variance"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 15,
      "solve_status": "FAIL_minimum expected target variance value(after dilate) for this process is 0.457. Raise goal variance"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 16,
      "solve_status": "FAIL_Co-Linearity found with assets [['000001.SH' '000002.SH']]"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 17,
      "solve_status": "FAIL_Co-Linearity found with assets [['000001.SH' '000002.SH']]"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 18,
      "solve_status": "FAIL_Co-Linearity found with assets [['000001.SH' '000002.SH']]"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0,
      "portf_std": -1.0,
      "portf_var": -0.01,
      "portf_w": [],
      "position_no": 19,
      "solve_status": "FAIL_Co-Linearity found with assets [['000001.SH' '000002.SH']]"
    },
    {
      "assets_idlst": [
        "000001.SH",
        "000002.SH",
        "000009.SH",
        "000010.SH",
        "000016.SH"
      ],
      "portf_rtn": 0.0023426334836694402,
      "portf_std": 0.005477225683874355,
      "portf_var": 3.0000001192092897e-05,
      "portf_w": [
        12.244567289856773,
        -6.654390028800006,
        -1.4713023803676504,
        -3.8811949702690027,
        0.7623202355613956
      ],
      "position_no": 20,
      "solve_status": "direct"
    }
  ],
  "weights": [
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      0.2,
      0.2,
      0.2,
      0.2,
      0.2
    ],
    [
      12.244567289856773,
      -6.654390028800006,
      -1.4713023803676504,
      -3.8811949702690027,
      0.7623202355613956
    ]
  ]
}


import warnings
from flask import request, Blueprint

warnings.filterwarnings('ignore')
app_name = __name__
static_folder = "Static"
template_folder = 'Template'

mvopt_api = Blueprint('mean_var', __name__)

@mvopt_api.route('/asset_allocate/mean_var_opt', methods=['POST'])
def runner():
    return result