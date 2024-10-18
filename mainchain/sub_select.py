from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd

def atom2bond(mol, atom_begin, atom_end) -> int:
    """返回两个原子之间的键所引
 
    :param mol: rdkit.Chem.rdchem.Mol, 需要处理的分子
    :param atom_begin: int, 起始原子
    :param atom_end: int, 结束原子
    :return: int, 键索引
    """

    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() == atom_begin and bond.GetEndAtomIdx() == atom_end:
            return bond.GetIdx()
        if bond.GetBeginAtomIdx() == atom_end and bond.GetEndAtomIdx() == atom_begin:
            return bond.GetIdx()
    return None

def get_substructure_smiles(mol, root_atom, radius) -> str:
    """返回子结构的SMILES字符串
 
    :param mol: rdkit.Chem.rdchem.Mol, 需要处理的分子
    :param root_atom: int, 中心原子索引
    :param radius: int, 子结构半径
    :return: str, 子结构的SMILES字符串
    """
    # 更新
    # 现在不仅需要输出SMILES字符串，还需要输出子结构中原子和键的索引

    atom_list = []
    bond_list = []
    if radius > 0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, root_atom)
        atom_map = {}
        _ = Chem.PathToSubmol(mol, env, atomMap=atom_map)
        # 获取子结构原子索引和键索引
        for atom in atom_map:
            atom_list.append(atom)

        for atom in atom_list:
            for bond in mol.GetBonds():
                if bond.GetBeginAtomIdx() == atom and bond.GetEndAtomIdx() in atom_list:
                    bond_list.append(bond.GetIdx())

        substructure_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_map, allBondsExplicit=True, allHsExplicit=True)
    else:
        atom_list.append(root_atom)
        substructure_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=(root_atom, ), allBondsExplicit=True, allHsExplicit=True)
    
    return substructure_smiles, atom_list, bond_list

def random_SMILES(mol) -> str:
    """ 生成当前分子的随机SMILES字符串
    
    :param mol: rdkit.Chem.rdchem.Mol, 需要处理的分子
    :return: str, 随机SMILES字符串
    """
    random_smi = Chem.MolToSmiles(mol,doRandom=True,canonical=False, allBondsExplicit=True, allHsExplicit=True)
    return random_smi

def upgrade_structure(mol, root_atom, radius) -> str:
    """ 子结构的父结构SMILES
    
    :param mol: rdkit.Chem.rdchem.Mol, 需要处理的分子
    :param root_atom: int, 中心原子索引
    :param radius: int, 子结构半径
    :return: str, 父结构SMILES
    """
    
    smi = get_substructure_smiles(mol, root_atom, radius + 1)
    return smi

def is_aromatic(mol, atom_list) -> bool:
    """ 判断子结构里是否有芳香原子

    :param mol: rdkit.Chem.rdchem.Mol, 分子对象
    :param atom_list: list, 子结构原子索引
    :return: bool, 有无芳香原子
    """
    
    for atom in atom_list:
        if mol.GetAtomWithIdx(atom).GetIsAromatic():
            return True
    return False

def is_valid_smiles(smiles) -> bool:
    """ 判断一个SMILES字符串是否是合法分子

    :param smiles: str, 需要处理的分子
    :return: bool, 是否是合法分子
    """
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return False
        else:
            return True
    except Exception:
        return False

def is_repeat_smiles(smiles, final_sub) -> bool:
    """ 判断一个子结构的SMILES字符串是否已经出现过/已选择
    
    :param smiles: str, 子结构的SMILES字符串
    :param final_sub: list, 已选子结构信息
    :return: bool, 是否重复
    """
    
    smiles_list = []
    for sub in final_sub:
        smiles_list.append(sub[0])

    if smiles in smiles_list:
        return True
    else:
        return False

def complete(mol, root_atom, radius) -> tuple:
    """ 将一个非法子结构补全(修复未闭合的苯环)

    :param mol: rdkit.Chem.rdchem.Mol, 分子对象
    :param root_atom: int, 中心原子索引
    :param radius: int, 当前子结构半径
    :return: tuple, 补全后的子结构信息
    """
    
    if radius > 0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, root_atom)
        atom_map = {}
        _ = Chem.PathToSubmol(mol, env, atomMap=atom_map)
    else:
        atom_map = (root_atom,)


    atom_to_add = set() # set of atom index
    for atom in atom_map: # collect current substructure atom index
        atom_to_add.add(atom)
    
    ri = mol.GetRingInfo()
    for atom in atom_map:
        for ring in ri.AtomRings():
            if atom in ring and mol.GetAtomWithIdx(atom).GetIsAromatic():
                for r in ring:
                    atom_to_add.add(r)
    
    complete_sub = Chem.MolFragmentToSmiles(mol, atom_to_add, allBondsExplicit=True, allHsExplicit=True)
    atom_list = list(atom_to_add)
    bond_list = []
    for atom in atom_list:
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() == atom and bond.GetEndAtomIdx() in atom_list:
                bond_list.append(bond.GetIdx())

    # check if the substructure is valid
    if not is_valid_smiles(complete_sub):
        print('Warning! Error happen, complete sub is still invalid:', Chem.MolToSmiles(mol))
        return None

    return complete_sub, atom_list, bond_list

def upgrade_structure(mol, root_atom, radius,) -> tuple:
    
    super_smiles, atom_list, bond_list = get_substructure_smiles(mol, root_atom, radius + 1)
    return super_smiles, atom_list, bond_list

def subselect(smiles, feature_importances, mfpgen, n) -> list:
    """子结构筛选算法-v1.2
 
    :param smiles: str, 分子结构的SMILES字符串
    :param feature_importances: DataFrame, 有RF训练得出的特征重要性
    :param mfpgen: xxx, 指纹生成器
    :param n: int, 选出的子结构个数
    :param radii: int, 摩根指纹半径
    :return final_sub: list, 选出的子结构
    """

    mol = Chem.MolFromSmiles(smiles)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    _ = mfpgen.GetCountFingerprint(mol,additionalOutput=ao)
    bitInfo = ao.GetBitInfoMap()

    subStructures = []
    for key, value in bitInfo.items():
        root_atom = value[0][0]
        radius = value[0][1]
        substructure_smiles, atom_lsit, bond_list = get_substructure_smiles(mol, root_atom, radius)
        subStructures.append((key, root_atom, radius, substructure_smiles, atom_lsit, bond_list))
    
    subStructures = sorted(subStructures, key=lambda x: feature_importances.loc[feature_importances['bit'] == x[0], 'importance'].values[0], reverse=True)

    final_sub = []
    i = 0
    while i < n:
        if i >= len(subStructures):
            final_sub.append((random_SMILES(mol), [-1], [-1]))
            i += 1
            continue
        
        root_atom = subStructures[i][1]
        radius = subStructures[i][2]
        sub_smiles = subStructures[i][3]
        atom_list = subStructures[i][4]
        bond_list = subStructures[i][5]

        if sub_smiles == '[CH2]':
            super_sub, _, _ = upgrade_structure(mol, root_atom, radius)
            # 这里跳过'[CH2]'的原因还需思考
            if '=' not in super_sub:
                i += 1
                n += 1
                continue
        
        if is_aromatic(mol, atom_list):
            complete_sub, atom_list, bond_list = complete(mol, root_atom, radius)
            if complete_sub == None:
                i += 1
                n += 1
                continue
            else:
                if is_repeat_smiles(complete_sub, final_sub):
                    n += 1
                else:
                    final_sub.append((complete_sub, atom_list, bond_list))
        else:
            if radius > 0:
                if is_repeat_smiles(sub_smiles, final_sub):
                    n += 1
                else:
                    final_sub.append((sub_smiles, atom_list, bond_list))
            else:
                if 'C' in sub_smiles or 'O' in sub_smiles or 'N' in sub_smiles:
                    super_sub, atom_list, bond_list = upgrade_structure(mol, root_atom, radius)
                    if is_valid_smiles(super_sub):
                        if is_repeat_smiles(super_sub, final_sub):
                            n += 1
                        else:
                            final_sub.append((super_sub, atom_list, bond_list))
                    else:
                        complete_sub, atom_list, bond_list = complete(mol, root_atom, radius+1)
                        if complete_sub == None:
                            i += 1
                            n += 1
                            continue
                        else:
                            if is_repeat_smiles(complete_sub, final_sub):
                                n += 1
                            else:
                                #--------------------------------
                                if is_aromatic(mol, atom_list):
                                    n += 1
                                #--------------------------------
                                else:
                                    final_sub.append((complete_sub, atom_list, bond_list))
                else:
                    final_sub.append((sub_smiles, atom_list, bond_list))
        
        i += 1
    final_sub_smiles = [sub[0] for sub in final_sub]
    final_sub_atom = [sub[1] for sub in final_sub]
    return final_sub_atom


# 加载数据集和特征重要性路径
df = pd.read_csv('../datasets/tg/tg.csv') # 需要重新修改数据集路径
df = df.dropna(subset=[df.columns[1]])
feature_importances = pd.read_csv('C:/Users/Blue/Desktop/数据科学研究组/Experiments/Substructures Enhanced MPNN/feature_importances/tg.csv') # 需要重新修改特征重要性路径，见文件夹Substructure Enhanced MPNN


n = 3
radii = 2
size = 8192
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radii,fpSize=size)


# 这里的'SMILES_k'需要根据n的值修改
df[['SMILES_1', 'SMILES_2', 'SMILES_3']]= df['SMILES'].apply(lambda x: pd.Series(subselect(x, feature_importances, mfpgen, n)))

property = df.columns[1]
property_column = df.pop(property)
df[property] = property_column


df.to_csv('tg_sub.csv', index=False) # 需要重新修改保存路径