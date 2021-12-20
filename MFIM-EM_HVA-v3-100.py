###############################################
# EM for HVA Ansatz - MFIM
###############################################

############################################
# Author - R Tali [rtali@iastate.edu]
# Version - v1.1
############################################


import torch
from sqlalchemy.sql.sqltypes import TIMESTAMP, Integer, Numeric
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table, Column, String, MetaData
from sqlalchemy import create_engine
import sqlalchemy as sa
import datetime as dt
import numpy as np
import random
from numpy import pi
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
from datetime import datetime
plt.style.use('fivethirtyeight')


if torch.cuda.is_available():

    print('Using MySQL on AWS to dump all data from the runs \n')

    continue_FLAG = True

    print('============================\n')
    print('If CUDA available? Yes')
    print('\n============================\n')

    # Define Device
    cuda0 = torch.device('cuda:0')

    ##################################
    # Logging in Database
    ##################################

    try:
        db = create_engine(
            'mysql+pymysql://vqe_user:Login1root@vqe.cuzj4xzkiyrf.us-east-2.rds.amazonaws.com:3306/Logs')
        base = declarative_base()
        Session = sessionmaker(db)
        session = Session()

    except Exception as e:

        print(e)

    ###################################################
    # Define ORM Adaptor
    ###################################################

    class EM_MODEL(base):
        __tablename__ = 'EM_MODEL'
        __table_args__ = {'schema': 'Logs', 'extend_existing': True}
        model_id = Column(Integer, primary_key=True)
        n_qubits = Column(Integer)
        g = Column(sa.Float)
        l = Column(Integer)
        ref = Column(String(10))

        # Overlap Type - 1

        o1t_0 = Column(sa.Float)
        o1t_1 = Column(sa.Float)
        o1t_2 = Column(sa.Float)
        o1t_3 = Column(sa.Float)
        o1t_4 = Column(sa.Float)
        o1t_5 = Column(sa.Float)
        o1t_6 = Column(sa.Float)
        o1t_7 = Column(sa.Float)
        o1t_8 = Column(sa.Float)
        o1t_9 = Column(sa.Float)

        o1t_10 = Column(sa.Float)
        o1t_11 = Column(sa.Float)
        o1t_12 = Column(sa.Float)
        o1t_13 = Column(sa.Float)
        o1t_14 = Column(sa.Float)
        o1t_15 = Column(sa.Float)
        o1t_16 = Column(sa.Float)
        o1t_17 = Column(sa.Float)
        o1t_18 = Column(sa.Float)
        o1t_19 = Column(sa.Float)

        o1t_20 = Column(sa.Float)
        o1t_21 = Column(sa.Float)
        o1t_22 = Column(sa.Float)
        o1t_23 = Column(sa.Float)
        o1t_24 = Column(sa.Float)
        o1t_25 = Column(sa.Float)
        o1t_26 = Column(sa.Float)
        o1t_27 = Column(sa.Float)
        o1t_28 = Column(sa.Float)
        o1t_29 = Column(sa.Float)

        o1t_30 = Column(sa.Float)
        o1t_31 = Column(sa.Float)
        o1t_32 = Column(sa.Float)
        o1t_33 = Column(sa.Float)
        o1t_34 = Column(sa.Float)
        o1t_35 = Column(sa.Float)
        o1t_36 = Column(sa.Float)
        o1t_37 = Column(sa.Float)
        o1t_38 = Column(sa.Float)
        o1t_39 = Column(sa.Float)

        o1t_40 = Column(sa.Float)
        o1t_41 = Column(sa.Float)
        o1t_42 = Column(sa.Float)
        o1t_43 = Column(sa.Float)
        o1t_44 = Column(sa.Float)
        o1t_45 = Column(sa.Float)
        o1t_46 = Column(sa.Float)
        o1t_47 = Column(sa.Float)
        o1t_48 = Column(sa.Float)
        o1t_49 = Column(sa.Float)

        o1t_50 = Column(sa.Float)
        o1t_51 = Column(sa.Float)
        o1t_52 = Column(sa.Float)
        o1t_53 = Column(sa.Float)
        o1t_54 = Column(sa.Float)
        o1t_55 = Column(sa.Float)
        o1t_56 = Column(sa.Float)
        o1t_57 = Column(sa.Float)
        o1t_58 = Column(sa.Float)
        o1t_59 = Column(sa.Float)

        o1t_60 = Column(sa.Float)
        o1t_61 = Column(sa.Float)
        o1t_62 = Column(sa.Float)
        o1t_63 = Column(sa.Float)
        o1t_64 = Column(sa.Float)
        o1t_65 = Column(sa.Float)
        o1t_66 = Column(sa.Float)
        o1t_67 = Column(sa.Float)
        o1t_68 = Column(sa.Float)
        o1t_69 = Column(sa.Float)

        o1t_70 = Column(sa.Float)
        o1t_71 = Column(sa.Float)
        o1t_72 = Column(sa.Float)
        o1t_73 = Column(sa.Float)
        o1t_74 = Column(sa.Float)
        o1t_75 = Column(sa.Float)
        o1t_76 = Column(sa.Float)
        o1t_77 = Column(sa.Float)
        o1t_78 = Column(sa.Float)
        o1t_79 = Column(sa.Float)

        o1t_80 = Column(sa.Float)
        o1t_81 = Column(sa.Float)
        o1t_82 = Column(sa.Float)
        o1t_83 = Column(sa.Float)
        o1t_84 = Column(sa.Float)
        o1t_85 = Column(sa.Float)
        o1t_86 = Column(sa.Float)
        o1t_87 = Column(sa.Float)
        o1t_88 = Column(sa.Float)
        o1t_89 = Column(sa.Float)

        o1t_90 = Column(sa.Float)
        o1t_91 = Column(sa.Float)
        o1t_92 = Column(sa.Float)
        o1t_93 = Column(sa.Float)
        o1t_94 = Column(sa.Float)
        o1t_95 = Column(sa.Float)
        o1t_96 = Column(sa.Float)
        o1t_97 = Column(sa.Float)
        o1t_98 = Column(sa.Float)
        o1t_99 = Column(sa.Float)

        o1t_100 = Column(sa.Float)
        o1t_101 = Column(sa.Float)
        o1t_102 = Column(sa.Float)
        o1t_103 = Column(sa.Float)
        o1t_104 = Column(sa.Float)
        o1t_105 = Column(sa.Float)
        o1t_106 = Column(sa.Float)
        o1t_107 = Column(sa.Float)
        o1t_108 = Column(sa.Float)
        o1t_109 = Column(sa.Float)

        o1t_110 = Column(sa.Float)
        o1t_111 = Column(sa.Float)
        o1t_112 = Column(sa.Float)
        o1t_113 = Column(sa.Float)
        o1t_114 = Column(sa.Float)
        o1t_115 = Column(sa.Float)
        o1t_116 = Column(sa.Float)
        o1t_117 = Column(sa.Float)
        o1t_118 = Column(sa.Float)
        o1t_119 = Column(sa.Float)

        o1t_120 = Column(sa.Float)
        o1t_121 = Column(sa.Float)
        o1t_122 = Column(sa.Float)
        o1t_123 = Column(sa.Float)
        o1t_124 = Column(sa.Float)
        o1t_125 = Column(sa.Float)
        o1t_126 = Column(sa.Float)
        o1t_127 = Column(sa.Float)
        o1t_128 = Column(sa.Float)
        o1t_129 = Column(sa.Float)

        o1t_130 = Column(sa.Float)
        o1t_131 = Column(sa.Float)
        o1t_132 = Column(sa.Float)
        o1t_133 = Column(sa.Float)
        o1t_134 = Column(sa.Float)
        o1t_135 = Column(sa.Float)
        o1t_136 = Column(sa.Float)
        o1t_137 = Column(sa.Float)
        o1t_138 = Column(sa.Float)
        o1t_139 = Column(sa.Float)

        o1t_140 = Column(sa.Float)
        o1t_141 = Column(sa.Float)
        o1t_142 = Column(sa.Float)
        o1t_143 = Column(sa.Float)
        o1t_144 = Column(sa.Float)
        o1t_145 = Column(sa.Float)
        o1t_146 = Column(sa.Float)
        o1t_147 = Column(sa.Float)
        o1t_148 = Column(sa.Float)
        o1t_149 = Column(sa.Float)

        o1t_150 = Column(sa.Float)
        o1t_151 = Column(sa.Float)
        o1t_152 = Column(sa.Float)
        o1t_153 = Column(sa.Float)
        o1t_154 = Column(sa.Float)
        o1t_155 = Column(sa.Float)
        o1t_156 = Column(sa.Float)
        o1t_157 = Column(sa.Float)
        o1t_158 = Column(sa.Float)
        o1t_159 = Column(sa.Float)

        o1t_160 = Column(sa.Float)
        o1t_161 = Column(sa.Float)
        o1t_162 = Column(sa.Float)
        o1t_163 = Column(sa.Float)
        o1t_164 = Column(sa.Float)
        o1t_165 = Column(sa.Float)
        o1t_166 = Column(sa.Float)
        o1t_167 = Column(sa.Float)
        o1t_168 = Column(sa.Float)
        o1t_169 = Column(sa.Float)

        o1t_170 = Column(sa.Float)
        o1t_171 = Column(sa.Float)
        o1t_172 = Column(sa.Float)
        o1t_173 = Column(sa.Float)
        o1t_174 = Column(sa.Float)
        o1t_175 = Column(sa.Float)
        o1t_176 = Column(sa.Float)
        o1t_177 = Column(sa.Float)
        o1t_178 = Column(sa.Float)
        o1t_179 = Column(sa.Float)

        o1t_180 = Column(sa.Float)
        o1t_181 = Column(sa.Float)
        o1t_182 = Column(sa.Float)
        o1t_183 = Column(sa.Float)
        o1t_184 = Column(sa.Float)
        o1t_185 = Column(sa.Float)
        o1t_186 = Column(sa.Float)
        o1t_187 = Column(sa.Float)
        o1t_188 = Column(sa.Float)
        o1t_189 = Column(sa.Float)

        o1t_190 = Column(sa.Float)
        o1t_191 = Column(sa.Float)
        o1t_192 = Column(sa.Float)
        o1t_193 = Column(sa.Float)
        o1t_194 = Column(sa.Float)
        o1t_195 = Column(sa.Float)
        o1t_196 = Column(sa.Float)
        o1t_197 = Column(sa.Float)
        o1t_198 = Column(sa.Float)
        o1t_199 = Column(sa.Float)

        o1t_200 = Column(sa.Float)
        o1t_201 = Column(sa.Float)
        o1t_202 = Column(sa.Float)
        o1t_203 = Column(sa.Float)
        o1t_204 = Column(sa.Float)
        o1t_205 = Column(sa.Float)
        o1t_206 = Column(sa.Float)
        o1t_207 = Column(sa.Float)
        o1t_208 = Column(sa.Float)
        o1t_209 = Column(sa.Float)

        # Overlap Type - 2

        o2t_0 = Column(sa.Float)
        o2t_1 = Column(sa.Float)
        o2t_2 = Column(sa.Float)
        o2t_3 = Column(sa.Float)
        o2t_4 = Column(sa.Float)
        o2t_5 = Column(sa.Float)
        o2t_6 = Column(sa.Float)
        o2t_7 = Column(sa.Float)
        o2t_8 = Column(sa.Float)
        o2t_9 = Column(sa.Float)

        o2t_10 = Column(sa.Float)
        o2t_11 = Column(sa.Float)
        o2t_12 = Column(sa.Float)
        o2t_13 = Column(sa.Float)
        o2t_14 = Column(sa.Float)
        o2t_15 = Column(sa.Float)
        o2t_16 = Column(sa.Float)
        o2t_17 = Column(sa.Float)
        o2t_18 = Column(sa.Float)
        o2t_19 = Column(sa.Float)

        o2t_20 = Column(sa.Float)
        o2t_21 = Column(sa.Float)
        o2t_22 = Column(sa.Float)
        o2t_23 = Column(sa.Float)
        o2t_24 = Column(sa.Float)
        o2t_25 = Column(sa.Float)
        o2t_26 = Column(sa.Float)
        o2t_27 = Column(sa.Float)
        o2t_28 = Column(sa.Float)
        o2t_29 = Column(sa.Float)

        o2t_30 = Column(sa.Float)
        o2t_31 = Column(sa.Float)
        o2t_32 = Column(sa.Float)
        o2t_33 = Column(sa.Float)
        o2t_34 = Column(sa.Float)
        o2t_35 = Column(sa.Float)
        o2t_36 = Column(sa.Float)
        o2t_37 = Column(sa.Float)
        o2t_38 = Column(sa.Float)
        o2t_39 = Column(sa.Float)

        o2t_40 = Column(sa.Float)
        o2t_41 = Column(sa.Float)
        o2t_42 = Column(sa.Float)
        o2t_43 = Column(sa.Float)
        o2t_44 = Column(sa.Float)
        o2t_45 = Column(sa.Float)
        o2t_46 = Column(sa.Float)
        o2t_47 = Column(sa.Float)
        o2t_48 = Column(sa.Float)
        o2t_49 = Column(sa.Float)

        o2t_50 = Column(sa.Float)
        o2t_51 = Column(sa.Float)
        o2t_52 = Column(sa.Float)
        o2t_53 = Column(sa.Float)
        o2t_54 = Column(sa.Float)
        o2t_55 = Column(sa.Float)
        o2t_56 = Column(sa.Float)
        o2t_57 = Column(sa.Float)
        o2t_58 = Column(sa.Float)
        o2t_59 = Column(sa.Float)

        o2t_60 = Column(sa.Float)
        o2t_61 = Column(sa.Float)
        o2t_62 = Column(sa.Float)
        o2t_63 = Column(sa.Float)
        o2t_64 = Column(sa.Float)
        o2t_65 = Column(sa.Float)
        o2t_66 = Column(sa.Float)
        o2t_67 = Column(sa.Float)
        o2t_68 = Column(sa.Float)
        o2t_69 = Column(sa.Float)

        o2t_70 = Column(sa.Float)
        o2t_71 = Column(sa.Float)
        o2t_72 = Column(sa.Float)
        o2t_73 = Column(sa.Float)
        o2t_74 = Column(sa.Float)
        o2t_75 = Column(sa.Float)
        o2t_76 = Column(sa.Float)
        o2t_77 = Column(sa.Float)
        o2t_78 = Column(sa.Float)
        o2t_79 = Column(sa.Float)

        o2t_80 = Column(sa.Float)
        o2t_81 = Column(sa.Float)
        o2t_82 = Column(sa.Float)
        o2t_83 = Column(sa.Float)
        o2t_84 = Column(sa.Float)
        o2t_85 = Column(sa.Float)
        o2t_86 = Column(sa.Float)
        o2t_87 = Column(sa.Float)
        o2t_88 = Column(sa.Float)
        o2t_89 = Column(sa.Float)

        o2t_90 = Column(sa.Float)
        o2t_91 = Column(sa.Float)
        o2t_92 = Column(sa.Float)
        o2t_93 = Column(sa.Float)
        o2t_94 = Column(sa.Float)
        o2t_95 = Column(sa.Float)
        o2t_96 = Column(sa.Float)
        o2t_97 = Column(sa.Float)
        o2t_98 = Column(sa.Float)
        o2t_99 = Column(sa.Float)

        o2t_100 = Column(sa.Float)
        o2t_101 = Column(sa.Float)
        o2t_102 = Column(sa.Float)
        o2t_103 = Column(sa.Float)
        o2t_104 = Column(sa.Float)
        o2t_105 = Column(sa.Float)
        o2t_106 = Column(sa.Float)
        o2t_107 = Column(sa.Float)
        o2t_108 = Column(sa.Float)
        o2t_109 = Column(sa.Float)

        o2t_110 = Column(sa.Float)
        o2t_111 = Column(sa.Float)
        o2t_112 = Column(sa.Float)
        o2t_113 = Column(sa.Float)
        o2t_114 = Column(sa.Float)
        o2t_115 = Column(sa.Float)
        o2t_116 = Column(sa.Float)
        o2t_117 = Column(sa.Float)
        o2t_118 = Column(sa.Float)
        o2t_119 = Column(sa.Float)

        o2t_120 = Column(sa.Float)
        o2t_121 = Column(sa.Float)
        o2t_122 = Column(sa.Float)
        o2t_123 = Column(sa.Float)
        o2t_124 = Column(sa.Float)
        o2t_125 = Column(sa.Float)
        o2t_126 = Column(sa.Float)
        o2t_127 = Column(sa.Float)
        o2t_128 = Column(sa.Float)
        o2t_129 = Column(sa.Float)

        o2t_130 = Column(sa.Float)
        o2t_131 = Column(sa.Float)
        o2t_132 = Column(sa.Float)
        o2t_133 = Column(sa.Float)
        o2t_134 = Column(sa.Float)
        o2t_135 = Column(sa.Float)
        o2t_136 = Column(sa.Float)
        o2t_137 = Column(sa.Float)
        o2t_138 = Column(sa.Float)
        o2t_139 = Column(sa.Float)

        o2t_140 = Column(sa.Float)
        o2t_141 = Column(sa.Float)
        o2t_142 = Column(sa.Float)
        o2t_143 = Column(sa.Float)
        o2t_144 = Column(sa.Float)
        o2t_145 = Column(sa.Float)
        o2t_146 = Column(sa.Float)
        o2t_147 = Column(sa.Float)
        o2t_148 = Column(sa.Float)
        o2t_149 = Column(sa.Float)

        o2t_150 = Column(sa.Float)
        o2t_151 = Column(sa.Float)
        o2t_152 = Column(sa.Float)
        o2t_153 = Column(sa.Float)
        o2t_154 = Column(sa.Float)
        o2t_155 = Column(sa.Float)
        o2t_156 = Column(sa.Float)
        o2t_157 = Column(sa.Float)
        o2t_158 = Column(sa.Float)
        o2t_159 = Column(sa.Float)

        o2t_160 = Column(sa.Float)
        o2t_161 = Column(sa.Float)
        o2t_162 = Column(sa.Float)
        o2t_163 = Column(sa.Float)
        o2t_164 = Column(sa.Float)
        o2t_165 = Column(sa.Float)
        o2t_166 = Column(sa.Float)
        o2t_167 = Column(sa.Float)
        o2t_168 = Column(sa.Float)
        o2t_169 = Column(sa.Float)

        o2t_170 = Column(sa.Float)
        o2t_171 = Column(sa.Float)
        o2t_172 = Column(sa.Float)
        o2t_173 = Column(sa.Float)
        o2t_174 = Column(sa.Float)
        o2t_175 = Column(sa.Float)
        o2t_176 = Column(sa.Float)
        o2t_177 = Column(sa.Float)
        o2t_178 = Column(sa.Float)
        o2t_179 = Column(sa.Float)

        o2t_180 = Column(sa.Float)
        o2t_181 = Column(sa.Float)
        o2t_182 = Column(sa.Float)
        o2t_183 = Column(sa.Float)
        o2t_184 = Column(sa.Float)
        o2t_185 = Column(sa.Float)
        o2t_186 = Column(sa.Float)
        o2t_187 = Column(sa.Float)
        o2t_188 = Column(sa.Float)
        o2t_189 = Column(sa.Float)

        o2t_190 = Column(sa.Float)
        o2t_191 = Column(sa.Float)
        o2t_192 = Column(sa.Float)
        o2t_193 = Column(sa.Float)
        o2t_194 = Column(sa.Float)
        o2t_195 = Column(sa.Float)
        o2t_196 = Column(sa.Float)
        o2t_197 = Column(sa.Float)
        o2t_198 = Column(sa.Float)
        o2t_199 = Column(sa.Float)

        o2t_200 = Column(sa.Float)
        o2t_201 = Column(sa.Float)
        o2t_202 = Column(sa.Float)
        o2t_203 = Column(sa.Float)
        o2t_204 = Column(sa.Float)
        o2t_205 = Column(sa.Float)
        o2t_206 = Column(sa.Float)
        o2t_207 = Column(sa.Float)
        o2t_208 = Column(sa.Float)
        o2t_209 = Column(sa.Float)

    # Energy

        e_0 = Column(sa.Float)
        e_1 = Column(sa.Float)
        e_2 = Column(sa.Float)
        e_3 = Column(sa.Float)
        e_4 = Column(sa.Float)
        e_5 = Column(sa.Float)
        e_6 = Column(sa.Float)
        e_7 = Column(sa.Float)
        e_8 = Column(sa.Float)
        e_9 = Column(sa.Float)

        e_10 = Column(sa.Float)
        e_11 = Column(sa.Float)
        e_12 = Column(sa.Float)
        e_13 = Column(sa.Float)
        e_14 = Column(sa.Float)
        e_15 = Column(sa.Float)
        e_16 = Column(sa.Float)
        e_17 = Column(sa.Float)
        e_18 = Column(sa.Float)
        e_19 = Column(sa.Float)

        e_20 = Column(sa.Float)
        e_21 = Column(sa.Float)
        e_22 = Column(sa.Float)
        e_23 = Column(sa.Float)
        e_24 = Column(sa.Float)
        e_25 = Column(sa.Float)
        e_26 = Column(sa.Float)
        e_27 = Column(sa.Float)
        e_28 = Column(sa.Float)
        e_29 = Column(sa.Float)

        e_30 = Column(sa.Float)
        e_31 = Column(sa.Float)
        e_32 = Column(sa.Float)
        e_33 = Column(sa.Float)
        e_34 = Column(sa.Float)
        e_35 = Column(sa.Float)
        e_36 = Column(sa.Float)
        e_37 = Column(sa.Float)
        e_38 = Column(sa.Float)
        e_39 = Column(sa.Float)

        e_40 = Column(sa.Float)
        e_41 = Column(sa.Float)
        e_42 = Column(sa.Float)
        e_43 = Column(sa.Float)
        e_44 = Column(sa.Float)
        e_45 = Column(sa.Float)
        e_46 = Column(sa.Float)
        e_47 = Column(sa.Float)
        e_48 = Column(sa.Float)
        e_49 = Column(sa.Float)

        e_50 = Column(sa.Float)
        e_51 = Column(sa.Float)
        e_52 = Column(sa.Float)
        e_53 = Column(sa.Float)
        e_54 = Column(sa.Float)
        e_55 = Column(sa.Float)
        e_56 = Column(sa.Float)
        e_57 = Column(sa.Float)
        e_58 = Column(sa.Float)
        e_59 = Column(sa.Float)

        e_60 = Column(sa.Float)
        e_61 = Column(sa.Float)
        e_62 = Column(sa.Float)
        e_63 = Column(sa.Float)
        e_64 = Column(sa.Float)
        e_65 = Column(sa.Float)
        e_66 = Column(sa.Float)
        e_67 = Column(sa.Float)
        e_68 = Column(sa.Float)
        e_69 = Column(sa.Float)

        e_70 = Column(sa.Float)
        e_71 = Column(sa.Float)
        e_72 = Column(sa.Float)
        e_73 = Column(sa.Float)
        e_74 = Column(sa.Float)
        e_75 = Column(sa.Float)
        e_76 = Column(sa.Float)
        e_77 = Column(sa.Float)
        e_78 = Column(sa.Float)
        e_79 = Column(sa.Float)

        e_80 = Column(sa.Float)
        e_81 = Column(sa.Float)
        e_82 = Column(sa.Float)
        e_83 = Column(sa.Float)
        e_84 = Column(sa.Float)
        e_85 = Column(sa.Float)
        e_86 = Column(sa.Float)
        e_87 = Column(sa.Float)
        e_88 = Column(sa.Float)
        e_89 = Column(sa.Float)

        e_90 = Column(sa.Float)
        e_91 = Column(sa.Float)
        e_92 = Column(sa.Float)
        e_93 = Column(sa.Float)
        e_94 = Column(sa.Float)
        e_95 = Column(sa.Float)
        e_96 = Column(sa.Float)
        e_97 = Column(sa.Float)
        e_98 = Column(sa.Float)
        e_99 = Column(sa.Float)

        e_100 = Column(sa.Float)
        e_101 = Column(sa.Float)
        e_102 = Column(sa.Float)
        e_103 = Column(sa.Float)
        e_104 = Column(sa.Float)
        e_105 = Column(sa.Float)
        e_106 = Column(sa.Float)
        e_107 = Column(sa.Float)
        e_108 = Column(sa.Float)
        e_109 = Column(sa.Float)

        e_110 = Column(sa.Float)
        e_111 = Column(sa.Float)
        e_112 = Column(sa.Float)
        e_113 = Column(sa.Float)
        e_114 = Column(sa.Float)
        e_115 = Column(sa.Float)
        e_116 = Column(sa.Float)
        e_117 = Column(sa.Float)
        e_118 = Column(sa.Float)
        e_119 = Column(sa.Float)

        e_120 = Column(sa.Float)
        e_121 = Column(sa.Float)
        e_122 = Column(sa.Float)
        e_123 = Column(sa.Float)
        e_124 = Column(sa.Float)
        e_125 = Column(sa.Float)
        e_126 = Column(sa.Float)
        e_127 = Column(sa.Float)
        e_128 = Column(sa.Float)
        e_129 = Column(sa.Float)

        e_130 = Column(sa.Float)
        e_131 = Column(sa.Float)
        e_132 = Column(sa.Float)
        e_133 = Column(sa.Float)
        e_134 = Column(sa.Float)
        e_135 = Column(sa.Float)
        e_136 = Column(sa.Float)
        e_137 = Column(sa.Float)
        e_138 = Column(sa.Float)
        e_139 = Column(sa.Float)

        e_140 = Column(sa.Float)
        e_141 = Column(sa.Float)
        e_142 = Column(sa.Float)
        e_143 = Column(sa.Float)
        e_144 = Column(sa.Float)
        e_145 = Column(sa.Float)
        e_146 = Column(sa.Float)
        e_147 = Column(sa.Float)
        e_148 = Column(sa.Float)
        e_149 = Column(sa.Float)

        e_150 = Column(sa.Float)
        e_151 = Column(sa.Float)
        e_152 = Column(sa.Float)
        e_153 = Column(sa.Float)
        e_154 = Column(sa.Float)
        e_155 = Column(sa.Float)
        e_156 = Column(sa.Float)
        e_157 = Column(sa.Float)
        e_158 = Column(sa.Float)
        e_159 = Column(sa.Float)

        e_160 = Column(sa.Float)
        e_161 = Column(sa.Float)
        e_162 = Column(sa.Float)
        e_163 = Column(sa.Float)
        e_164 = Column(sa.Float)
        e_165 = Column(sa.Float)
        e_166 = Column(sa.Float)
        e_167 = Column(sa.Float)
        e_168 = Column(sa.Float)
        e_169 = Column(sa.Float)

        e_170 = Column(sa.Float)
        e_171 = Column(sa.Float)
        e_172 = Column(sa.Float)
        e_173 = Column(sa.Float)
        e_174 = Column(sa.Float)
        e_175 = Column(sa.Float)
        e_176 = Column(sa.Float)
        e_177 = Column(sa.Float)
        e_178 = Column(sa.Float)
        e_179 = Column(sa.Float)

        e_180 = Column(sa.Float)
        e_181 = Column(sa.Float)
        e_182 = Column(sa.Float)
        e_183 = Column(sa.Float)
        e_184 = Column(sa.Float)
        e_185 = Column(sa.Float)
        e_186 = Column(sa.Float)
        e_187 = Column(sa.Float)
        e_188 = Column(sa.Float)
        e_189 = Column(sa.Float)

        e_190 = Column(sa.Float)
        e_191 = Column(sa.Float)
        e_192 = Column(sa.Float)
        e_193 = Column(sa.Float)
        e_194 = Column(sa.Float)
        e_195 = Column(sa.Float)
        e_196 = Column(sa.Float)
        e_197 = Column(sa.Float)
        e_198 = Column(sa.Float)
        e_199 = Column(sa.Float)

        e_200 = Column(sa.Float)
        e_201 = Column(sa.Float)
        e_202 = Column(sa.Float)
        e_203 = Column(sa.Float)
        e_204 = Column(sa.Float)
        e_205 = Column(sa.Float)
        e_206 = Column(sa.Float)
        e_207 = Column(sa.Float)
        e_208 = Column(sa.Float)
        e_209 = Column(sa.Float)

    class Entry(base):
        __tablename__ = 'iterlog'
        __table_args__ = {'schema': 'Logs', 'extend_existing': True}
        logid = Column(Integer, primary_key=True)
        owner = Column(String(1))
        n_qubits = Column(Integer)
        g = Column(sa.Float)
        layers = Column(Integer)
        ETA = Column(sa.Float)
        MAX_ITER = Column(Integer)
        init_type = Column(String(5))
        NUM_ROUNDS = Column(Integer)
        iter = Column(Integer)
        overlap = Column(sa.Float)
        energy = Column(sa.Float)
        norm_grad = Column(sa.Float)
        vector = Column(String(4))
        angles = Column(String(200))
        log_start_time = Column(TIMESTAMP)
        atype = Column(String(10))

    class Round(base):
        __tablename__ = 'roundlog'
        __table_args__ = {'schema': 'Logs', 'extend_existing': True}
        logid = Column(Integer, primary_key=True)
        owner = Column(String(1))
        n_qubits = Column(Integer)
        g = Column(sa.Float)
        layers = Column(Integer)
        ETA = Column(sa.Float)
        MAX_ITER = Column(Integer)
        init_type = Column(String(5))
        NUM_ROUNDS = Column(Integer)
        round_id = Column(Integer)
        overlap = Column(sa.Float)
        energy = Column(sa.Float)
        ansatz = Column(String(4))
        params = Column(String(200))
        log_start_time = Column(TIMESTAMP)
        round_time = Column(sa.Float)
        atype = Column(String(10))

    class EMEntry(base):
        __tablename__ = 'emlog'
        __table_args__ = {'schema': 'Logs', 'extend_existing': True}
        logid = Column(Integer, primary_key=True)
        owner = Column(String(1))
        n_qubits = Column(Integer)
        g = Column(sa.Float)
        layers = Column(Integer)
        ETA = Column(sa.Float)
        MAX_ITER = Column(Integer)
        init_type = Column(String(5))
        iter = Column(Integer)
        overlap = Column(sa.Float)
        energy = Column(sa.Float)
        norm_grad = Column(sa.Float)
        vector = Column(String(4))
        angles = Column(String(200))
        log_start_time = Column(TIMESTAMP)
        atype = Column(String(10))

    class EMSummary(base):
        __tablename__ = 'emsummary'
        __table_args__ = {'schema': 'Logs', 'extend_existing': True}
        logid = Column(Integer, primary_key=True)
        owner = Column(String(1))
        n_qubits = Column(Integer)
        g = Column(sa.Float)
        layers = Column(Integer)
        ETA = Column(sa.Float)
        MAX_ITER = Column(Integer)
        init_type = Column(String(5))
        NUMITERS = Column(Integer)
        overlap = Column(sa.Float)
        energy = Column(sa.Float)
        vector = Column(String(4))
        angles = Column(String(200))
        log_start_time = Column(TIMESTAMP)
        total_time = Column(sa.Float)
        atype = Column(String(10))

else:
    continue_FLAG = False

    print('============================\n')
    print('If CUDA available? No')
    print('\n Program will fail')
    print('\n============================\n')


##################################
# Define Basics
##################################

# single qubit basis states |0> and |1>
q0 = np.array([[1], [0]])
q1 = np.array([[0], [1]])

# Pauli Matrices
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
HG = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

######################
#  HELPER FUNCTIONS
######################


# Creates the all zero input state.
def all_Zero_State(n_qubits):
    if n_qubits < 2:
        return 'Invalid Input : Specify at least 2 qubits'

    else:
        # Init State
        init_all_zero = np.kron(q0, q0)

        for t in range(n_qubits - 2):
            init_all_zero = np.kron(init_all_zero, q0)

        return init_all_zero


# Creates Random Initial State Ansatz
def psi0(n_qubits):
    if n_qubits < 2:
        return 'Invalid Input : Specify at least 2 qubits'

    else:
        pick = random.uniform(0, 2 * pi)
        i1 = np.cos(pick) * q0 + np.sin(pick) * q1
        pick = random.uniform(0, 2 * pi)
        i2 = np.cos(pick) * q0 + np.sin(pick) * q1
        init_random = np.kron(i1, i2)

        for t in range(n_qubits - 2):
            pick = random.uniform(0, 2 * pi)
            inow = np.cos(pick) * q0 + np.sin(pick) * q1
            init_random = np.kron(init_random, inow)

        return init_random


# Creates the equally superimposed product state from all zero state.
def equal_Superposition(n_qubits, init_all_zero):
    if n_qubits < 2:
        return 'Invalid Input : Specify at least 2 qubits'

    else:

        all_H = np.kron(HG, HG)

        for t in range(n_qubits - 2):
            all_H = np.kron(all_H, HG)

        equal_Superpos = all_H @ init_all_zero

        return equal_Superpos


# Analytical Ground State using Numpy's inbuilt eigh function.
def get_analytical_ground_state(H):
    e, v = LA.eigh(H)
    return np.min(e), v[:, np.argmin(e)]


# Create Unitary
def CU(Q, theta, n_qubits):
    Id = np.eye(2 ** n_qubits)
    return np.cos(theta) * Id - 1j * np.sin(theta) * Q


# Create Unitary within CUDA
def CUTensor(Q, theta, n_qubits):
    Id = torch.eye(2 ** n_qubits, dtype=torch.cdouble, device=cuda0)
    U = np.cos(theta) * Id - 1j * np.sin(theta) * \
        torch.tensor(Q, dtype=torch.cdouble, device=cuda0)
    return U


# Ansatz - MFIM
def ansatz_vha(
        X_param_set,
        ZZ_param_set,
        Z_param_set,
        components,
        n_qubits,
        layers):
    # Initialize Ansatz to I
    ansatz = np.eye(2 ** n_qubits)

    ZZ_components = components[0]
    X_components = components[1]
    Z_components = components[2]

    for layer in range(layers):

        for ct1, comp1 in enumerate(ZZ_components):
            ansatz = CU(
                comp1,
                theta=ZZ_param_set[layer],
                n_qubits=n_qubits) @ ansatz

        for ct2, comp2 in enumerate(X_components):
            ansatz = CU(
                comp2,
                theta=X_param_set[layer],
                n_qubits=n_qubits) @ ansatz

        for ct3, comp3 in enumerate(Z_components):
            ansatz = CU(
                comp3,
                theta=Z_param_set[layer],
                n_qubits=n_qubits) @ ansatz

    return ansatz


##################################
# Define MFIM model
##################################

# helper funnction for TFIM model creation.
def component_sums(components, n_qubits):
    ZZ_sum = np.zeros((2 ** n_qubits, 2 ** n_qubits))
    X_sum = np.zeros((2 ** n_qubits, 2 ** n_qubits))
    Z_sum = np.zeros((2 ** n_qubits, 2 ** n_qubits))

    for zz_arr in components[0]:
        ZZ_sum += zz_arr

    for x_arr in components[1]:
        X_sum += x_arr

    for z_arr in components[2]:
        Z_sum += z_arr

    return ZZ_sum, X_sum, Z_sum


# Coding
def array_coding_to_kron(arr, type):
    n_qubits = len(arr)

    if type == 'ZZ':
        convert = {0: I, 1: Z}  # Dictionary that maps code to Pauli Matrix
        expr = np.kron(convert[arr[0]], convert[arr[1]])
        for t in range(2, n_qubits):
            expr = np.kron(expr, convert[arr[t]])

        return expr

    else:
        if type == 'X':
            convert = {0: I, 1: X}
            expr2 = np.kron(convert[arr[0]], convert[arr[1]])
            for k in range(2, n_qubits):
                expr2 = np.kron(expr2, convert[arr[k]])

            return expr2
        else:
            convert = {0: I, 1: Z}
            expr3 = np.kron(convert[arr[0]], convert[arr[1]])
            for m in range(2, n_qubits):
                expr3 = np.kron(expr3, convert[arr[m]])

            return expr3


# MFIM
def create_MFIM(n_qubits, g):
    if n_qubits == 2:
        return -1 * np.kron(Z, Z) - g * (np.kron(X, I) + np.kron(I, X)) - g * (np.kron(Z, I) + np.kron(I, Z)), {
            0: [np.kron(Z, Z)],
            1: [np.kron(X, I), np.kron(I, X)],
            2: [np.kron(Z, I), np.kron(I, Z)]
        }

    else:
        # This will store all the kronecker products used in Ansatz Layers
        comps = {0: [], 1: [], 2: []}

        # Initializing an empty
        mfim = np.zeros((2 ** n_qubits, 2 ** n_qubits))

        # Encode ZZ Terms
        for i in range(n_qubits):
            zz_arr = np.zeros(n_qubits)
            if i < n_qubits - 1:
                zz_arr[i] = 1
                zz_arr[i + 1] = 1
            else:
                zz_arr[0] = 1
                zz_arr[i] = 1

            # Call the coding function
            mfim = mfim - array_coding_to_kron(zz_arr, type='ZZ')
            # Append component
            comps[0].append(array_coding_to_kron(zz_arr, type='ZZ'))

        # X Terms
        for i in range(n_qubits):
            x_arr = np.zeros(n_qubits)
            x_arr[i] = 1

            # Call the coding function
            mfim = mfim - g * array_coding_to_kron(x_arr, type='X')
            # Append component
            comps[1].append(array_coding_to_kron(x_arr, type='X'))

        # Z Terms
        for i in range(n_qubits):
            z_arr = np.zeros(n_qubits)
            z_arr[i] = 1

            # Call the coding function
            mfim = mfim - g * array_coding_to_kron(z_arr, type='Z')
            # Append component
            comps[2].append(array_coding_to_kron(z_arr, type='Z'))

        return mfim, comps


# This funaction calculates the overlap of the solution with analytical
# ground state.
def overlap_calculator(min_pm, ground_st):
    return np.abs(np.vdot(min_pm, ground_st)) ** 2


def power_computation(H, circuit_input):
    return (1 / (LA.norm(H @ circuit_input))) * (H @ circuit_input)


def energy_raw(H, psi):
    return np.real((psi.conj().T) @ H @ psi)[0][0]


# Helper functions to compute derivative

def all_X(X_components, param, n_qubits):
    X = torch.eye(2 ** n_qubits, dtype=torch.cdouble, device=cuda0)
    for component in X_components:
        X = CUTensor(component, param, n_qubits=n_qubits) @ X
    return X


def all_ZZ(ZZ_components, param, n_qubits):
    ZZ = torch.eye(2 ** n_qubits, dtype=torch.cdouble, device=cuda0)
    for component in ZZ_components:
        ZZ = CUTensor(component, param, n_qubits=n_qubits) @ ZZ
    return ZZ


def all_Z(Z_components, param, n_qubits):
    Z = torch.eye(2 ** n_qubits, dtype=torch.cdouble, device=cuda0)
    for component in Z_components:
        Z = CUTensor(component, param, n_qubits=n_qubits) @ Z
    return Z

# Send Component sums one time to GPU


def send_bulk_gpu(components, n_qubits, kind):
    try:

        # This is implemented via a function call.
        sum_ZZ, sum_X, sum_Z = component_sums(components, n_qubits=n_qubits)

        if kind == 'OM':
            # Send to CUDA
            cuda_sum_iZZ = torch.as_tensor(
                1j * sum_ZZ, dtype=torch.cdouble, device=cuda0)
            cuda_sum_iX = torch.as_tensor(
                1j * sum_X, dtype=torch.cdouble, device=cuda0)
            cuda_sum_iZ = torch.as_tensor(
                1j * sum_Z, dtype=torch.cdouble, device=cuda0)

            return cuda_sum_iZZ, cuda_sum_iX, cuda_sum_iZ
        else:
            cuda_sum_ZZ = torch.as_tensor(
                sum_ZZ, dtype=torch.cdouble, device=cuda0)
            cuda_sum_X = torch.as_tensor(
                sum_X, dtype=torch.cdouble, device=cuda0)
            cuda_sum_Z = torch.as_tensor(
                sum_Z, dtype=torch.cdouble, device=cuda0)

            return cuda_sum_ZZ, cuda_sum_X, cuda_sum_Z

    except Exception as e:
        print('\n ==== Error Sending to GPU===== \n')
        print(e)


def grad_positioning(grad):
    ZZ = []
    X = []
    Z = []
    for i in range(len(grad)):
        if i % 3 == 0:
            ZZ.append(grad[i])
        else:
            if i % 3 == 1:
                X.append(grad[i])
            else:
                Z.append(grad[i])

    return np.array(ZZ), np.array(X), np.array(Z)

##################

# EM

##################

# Gradient - Harrow Napp


def grad_harrow_napp(
        H,
        X_param_set,
        ZZ_param_set,
        Z_param_set,
        components,
        circuit_input,
        n_qubits,
        layers,
        cuda_ZZ,
        cuda_X,
        cuda_Z):
    # Prepare the common right hand side for the Harrow Napp Expression
    H_psi_right = H @ ansatz_vha(X_param_set=X_param_set,
                                 ZZ_param_set=ZZ_param_set,
                                 Z_param_set=Z_param_set,
                                 components=components,
                                 n_qubits=n_qubits,
                                 layers=layers) @ circuit_input

    # Send to GPU
    cuda_psi_rt = torch.as_tensor(
        H_psi_right,
        dtype=torch.cdouble,
        device=cuda0)

    # Sum the ZZ and X components
    # sum_ZZ, sum_X, sum_Z = component_sums(components, n_qubits=n_qubits)  #
    # This is implemented via a function call.

    # Total parameters
    param_per_layer = 3  # We always have 3 params per layer for VHA Ansatz.
    # This is just initialization for the gradient vector in CPU
    full_derivative = np.zeros(param_per_layer * layers)

    # Derivative Expression for each param

    # params

    # Loop through all layers
    for j in range(layers):
        # initialize computation for the jth ZZ derivative
        psi_left_d_ZZ = torch.as_tensor(
            circuit_input, dtype=torch.cdouble, device=cuda0)
        psi_left_d_X = torch.as_tensor(
            circuit_input, dtype=torch.cdouble, device=cuda0)
        psi_left_d_Z = torch.as_tensor(
            circuit_input, dtype=torch.cdouble, device=cuda0)

        # This inner loop is to loop through the circuit elements, only one of
        # the ZZ elements will have a derivative
        for i in range(layers):
            all_ZZs = all_ZZ(components[0], ZZ_param_set[i], n_qubits=n_qubits)
            all_Xs = all_X(components[1], X_param_set[i], n_qubits=n_qubits)
            all_Zs = all_Z(components[2], Z_param_set[i], n_qubits=n_qubits)

            if i == j:
                psi_left_d_ZZ = all_Zs @ all_Xs @ all_ZZs @ cuda_ZZ @ psi_left_d_ZZ
                psi_left_d_X = all_Zs @ all_Xs @ cuda_X @ all_ZZs @ psi_left_d_X
                psi_left_d_Z = all_Zs @ all_Xs @ cuda_Z @ all_ZZs @ psi_left_d_Z
            else:
                psi_left_d_ZZ = all_Zs @ all_Xs @ all_ZZs @ psi_left_d_ZZ
                psi_left_d_X = all_Zs @ all_Xs @ all_ZZs @ psi_left_d_X
                psi_left_d_Z = all_Zs @ all_Xs @ all_ZZs @ psi_left_d_Z

        # Store
        full_derivative[j * param_per_layer] = -2 * torch.imag(
            (psi_left_d_ZZ.conj().T) @  cuda_psi_rt).cpu().detach().item()  # Sent to CPU
        full_derivative[j * param_per_layer + 1] = -2 * torch.imag(
            (psi_left_d_X.conj().T) @  cuda_psi_rt).cpu().detach().item()  # Sent to CPU
        full_derivative[j * param_per_layer + 2] = -2 * torch.imag(
            (psi_left_d_Z.conj().T) @  cuda_psi_rt).cpu().detach().item()  # Sent to CPU

    # Return all partial derivatives
    return full_derivative

# Function for Gradient Descent --> Vanilla variety.


def hn_grad_desc_quantum(
        ana_vector,
        H,
        components,
        X_param_set,
        ZZ_param_set,
        Z_param_set,
        circuit_input,
        MAXITERS,
        eta,
        GRADTOL,
        n_qubits,
        layers,
        cuda_ZZ,
        cuda_X,
        cuda_Z,
        time_start,
        plotting='off',
        logging='on',
        log_freq=1):
    store_grad_norm = []
    store_energy = []
    store_vector = []

    # Theta is a vector ---> np.array
    theta_X = X_param_set.copy()
    theta_ZZ = ZZ_param_set.copy()
    theta_Z = Z_param_set.copy()

    # Keep track of number of iterations
    counter = 0

    # Iterate
    for iter in range(MAXITERS):

        grad = grad_harrow_napp(
            H=H,
            X_param_set=theta_X,
            ZZ_param_set=theta_ZZ,
            Z_param_set=theta_Z,
            components=components,
            circuit_input=circuit_input,
            n_qubits=n_qubits,
            layers=layers,
            cuda_ZZ=cuda_ZZ,
            cuda_X=cuda_X,
            cuda_Z=cuda_Z)

        if LA.norm(grad) < GRADTOL:
            break

        # Extract components - This is to correctly order gradient components
        ZZ, X, Z = grad_positioning(grad)

        # Update thetas
        theta_ZZ = theta_ZZ - eta * ZZ
        theta_X = theta_X - eta * X
        theta_Z = theta_Z - eta * Z

        # Eigenvector
        v = ansatz_vha(
            X_param_set=theta_X,
            ZZ_param_set=theta_ZZ,
            Z_param_set=theta_Z,
            components=components,
            n_qubits=n_qubits,
            layers=layers) @ circuit_input

        # Overlap
        ov = np.abs(np.vdot(v, ana_vector)) ** 2

        # Energy
        e = energy_raw(H, v)

        # Some Periodic Logging on Terminal for large N --> if requested.
        if logging == 'on':
            # Log every 20 steps.
            if counter % log_freq == 0:
                vals_now = EMEntry(owner='R',
                                   n_qubits=n_qubits,
                                   g=g,
                                   layers=layers,
                                   ETA=eta,
                                   MAX_ITER=MAXITERS,
                                   init_type='eq',
                                   iter=counter,
                                   overlap=ov,
                                   energy=e,
                                   norm_grad=LA.norm(grad),
                                   vector='none',
                                   angles=str([theta_ZZ,
                                               theta_X,
                                               theta_Z]),
                                   log_start_time=time_start,
                                   atype='m-hva-mfim')

                session.add(vals_now)
                session.commit()

        # Store Gradient Norm and Energy
        store_grad_norm.append(LA.norm(grad))
        store_energy.append(e)
        store_vector.append(v)

        # Keep track of number of iterations
        counter += 1

    # Some Plotting --> if requested.
    if plotting == 'on':
        plt.plot(range(counter), store_grad_norm)
        plt.title('Track Gradient Norm')
        plt.xlabel('Iteration Number')
        plt.ylabel('L2 Norm of the Gradient')
        plt.show()

        plt.plot(range(counter), store_energy)
        plt.title('Track Cost Function')
        plt.xlabel('Iteration Number')
        plt.ylabel('Minimum Eigen Value attained')
        plt.show()

    return [theta_ZZ, theta_X, theta_Z], counter, v, LA.norm(
        grad), store_vector, store_energy


def serial_statistics(session, n, g, l, v_hist, e_hist):

    v_final = v_hist[99]

    v_olap_t1 = []
    v_olap_t2 = []
    e = []

    for i in range(210):
        v_olap_t1.append(overlap_calculator(v_hist[i], v_final))
        e.append(e_hist[i])
        if i < 209:
            v_olap_t2.append(overlap_calculator(
                v_hist[i], v_hist[i + 1]))
        else:
            v_olap_t2.append(1)

    # Write everything to Database
    insert_data = {'n_qubits': n, 'g': g, 'l': l, 'ref': 'EM-MOD100'}
    for h in range(len(v_olap_t1)):
        insert_data['o1t_' + str(h)] = v_olap_t1[h]
        insert_data['o2t_' + str(h)] = v_olap_t2[h]
        insert_data['e_' + str(h)] = e[h]

    em_model_data = EM_MODEL(**insert_data)
    session.add(em_model_data)
    session.commit()


if continue_FLAG:

    MAX_ITERS = 210  # Enter this at the start of the Program
    ETA = 0.005  # Enter this at the start of the Program

    # Update as necessary - Only used for Overlap Maximization
    OFFSET_DICT = {4: 15, 6: 30, 8: 50, 10: 100}
    LAYERS_DICT = {4: [2, 4, 6], 6: [2, 4, 6, 8], 8: [2, 4, 6, 8, 10], 10: [
        2, 4, 6, 8, 10, 12]}  # Update as necessary - Specifying the layers for MFIM Model
    OWNER = 'R'  # For Justin = 'J', For Ronak = 'R'
    # .strftime('%Y-%m-%d %H:%M:%S') #If using Postgres instead of SQLite then uncomment the strftime part.
    tstart = dt.datetime.now()
    # Important - We identify a run by this value. Save this somewhere.
    print(tstart)

    for n_qubit in [4,6]:
        print('MFIM - At n_qubit = ', n_qubit)
        #for g in [0.5, 1.0, 2.0]:
        for g in np.linspace(0.1,2.0,150):

            # Create the MFIM Model
            H, components = create_MFIM(n_qubits=n_qubit, g=g)

            # Diagonalize
            e_an, v_an = get_analytical_ground_state(H)

            # Send components to GPU
            SZZ, SX, SZ = send_bulk_gpu(components, n_qubit, kind='EM')

            print('At g = ', g)

            for layer in LAYERS_DICT[n_qubit]:
                print('At Layer = ', layer)

                X_param_set = (pi / 3) * np.ones(layer)
                ZZ_param_set = (pi / 3) * np.ones(layer)
                Z_param_set = (pi / 3) * np.ones(layer)

                circuit_input = equal_Superposition(
                    n_qubit, all_Zero_State(n_qubit))

                round_start_time = time.time()
                # Call Gradient Descent
                h_theta, cnt, h_eigen_vector, h_grad, v_hist, e_hist = hn_grad_desc_quantum(ana_vector=v_an,
                                                                                            H=H,
                                                                                            components=components,
                                                                                            X_param_set=X_param_set,
                                                                                            ZZ_param_set=ZZ_param_set,
                                                                                            Z_param_set=Z_param_set,
                                                                                            circuit_input=circuit_input,
                                                                                            MAXITERS=MAX_ITERS,
                                                                                            eta=ETA,
                                                                                            GRADTOL=0.00001,
                                                                                            n_qubits=n_qubit,
                                                                                            layers=layer,
                                                                                            cuda_ZZ=SZZ,
                                                                                            cuda_X=SX,
                                                                                            cuda_Z=SZ,
                                                                                            time_start=tstart,
                                                                                            plotting='off',
                                                                                            logging='on',
                                                                                            log_freq=1)

                round_end_time = time.time()
                round_time = (round_end_time - round_start_time) / 60

                #print('EM - Gradient descent time for 1 at g = ', g, ' , layers = ', layer, ' = ', round_time, ' mins')

                # Overlap
                olap = np.abs(np.vdot(h_eigen_vector, v_an)) ** 2

                # Energy
                energy = energy_raw(H, h_eigen_vector)

                # Print overlap
                print('Calculated Overlap after 500 steps of EM = ', olap)

                # Log at round level
                vals_round = EMSummary(
                    owner=OWNER,
                    n_qubits=n_qubit,
                    g=g,
                    layers=layer,
                    ETA=ETA,
                    MAX_ITER=MAX_ITERS,
                    init_type='eq',
                    NUMITERS=cnt,
                    overlap=olap,
                    energy=energy,
                    vector='none',
                    angles=str(h_theta),
                    log_start_time=tstart,
                    total_time=round_time,
                    atype='m-hva-mfim')

                session.add(vals_round)
                session.commit()

                # This is call for calculation and storage
                serial_statistics(session, n_qubit, g, layer, v_hist, e_hist)

    # Close Database Session
    session.close()


else:
    print('==================================================\n')
    print('Sorry no GPU available for use. PROGRAM TERMINATED\n')
    print('==================================================\n')
