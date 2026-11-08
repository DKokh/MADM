import os
import glob
import numpy as np
import pandas as pd
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io as strucio
import biotite.structure.io.pdb as bpdb
import hydride
from anarci import anarci
from anarci import number
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

import mdtraj as md
#-----------------------------------------------------------------------------------------

def three_to_one(three_letter_code):
    mapping = {'Aba':'A','Ace':'X','Acr':'X','Ala':'A','Aly':'K','Arg':'R','Asn':'N','Asp':'D','Cas':'C',
           'Ccs':'C','Cme':'C','Csd':'C','Cso':'C','Csx':'C','Cys':'C','Dal':'A','Dbb':'T','Dbu':'T',
           'Dha':'S','Gln':'Q','Glu':'E','Gly':'G','Glz':'G','His':'H','Hse':'S','Ile':'I','Leu':'L',
           'Llp':'K','Lys':'K','Men':'N','Met':'M','Mly':'K','Mse':'M','Nh2':'X','Nle':'L','Ocs':'C',
           'Pca':'E','Phe':'F','Pro':'P','Ptr':'Y','Sep':'S','Ser':'S','Thr':'T','Tih':'A','Tpo':'T',
           'Trp':'W','Tyr':'Y','Unk':'X','Val':'V','Ycm':'C','Sec':'U','Pyl':'O'} # you can add more
    return mapping[three_letter_code[0].upper() + three_letter_code[1:].lower()]
RES_CHARGE = {
    "ASP": -1.0, "GLU": -1.0,
    "LYS":  1.0, "ARG":  1.0,
    "HIS":  0.5,
}
AROMATIC = {'PHE', 'TYR', 'TRP', 'HIS'}
BACKBONE_ATOMS = {"N", "CA", "C", "O","OXT"}
# H-bond donors:  N-H, O-H, S-H groups — the heavy atom that carries H
HBOND_DONORS = {
    "SER":  ["OG"],
    "THR":  ["OG1"],
    "TYR":  ["OH"],
    "TRP":  ["NE1"],
    "ASN":  ["ND2"],
    "GLN":  ["NE2"],
    "LYS":  ["NZ"],
    "ARG":  ["NH1", "NH2", "NE"],
    "HIS":  ["ND1", "NE2"],
    "CYS":  ["SG"],
}

# H-bond acceptors: lone-pair O, N atoms
HBOND_ACCEPTORS = {
    "SER":  ["OG"],
    "THR":  ["OG1"],
    "TYR":  ["OH"],
    "ASN":  ["OD1"],
    "GLN":  ["OE1"],
    "ASP":  ["OD1", "OD2"],
    "GLU":  ["OE1", "OE2"],
    "HIS":  ["ND1", "NE2"],
    "MET":  ["SD"],
    "CYS":  ["SG"],
}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR"}
CHARGED = {"ARG", "LYS", "ASP", "GLU", "HIS"}

POSITIVE_ATOMS = { "LYS": ["NZ"], "ARG": ["NE", "NH1", "NH2"],}
NEGATIVE_ATOMS = { "ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"],}

POSITIVE = {"LYS", "ARG"}     # add "HIS" if you want it
NEGATIVE = {"ASP", "GLU"}


#---------
def rename_chain_a_to_hl(structure: struc.AtomArray) -> struc.AtomArray:
    """Rename chain 'A' to 'H' in-place on the chain_id annotation."""
    structure.chain_id[structure.chain_id == "A"] = "H"
    structure.chain_id[structure.chain_id == "B"] = "L"
    return structure

#--------------------
class ab_structure():
    def __init__(self, 
                 atom_array, 
                 REF_SASA,
                 HYDRO
                ):
        atom_array = rename_chain_a_to_hl(atom_array)
        
        if not bool(np.any(atom_array.element == "H")):
            atom_array, _ = protonation(atom_array)
            self.protonated = False
        else:
            self.protonated = True
        
        self.atom_array = atom_array
        heavy_mask = (
            struc.filter_amino_acids(self.atom_array) &
            (self.atom_array.element != "H")
        )
        self.heavy = self.atom_array[heavy_mask]
        self.com   = self.heavy.coord.mean(axis=0)          # (3,)

        self.residues  = list(struc.residue_iter(self.heavy))
        self.L         = len(self.residues)

        # per -residue parameters
        self.ca_coords  = np.zeros((self.L, 3))
        self.charges    = np.zeros(self.L)
        self.hydros     = np.zeros(self.L)
        self.hydro_exp  = np.zeros(self.L)
        self.hydro_full = np.zeros(self.L)
        self.charge_exp = np.zeros(self.L)
        self.exposures  = np.zeros(self.L)
        self.exposures_sch  = np.zeros(self.L)
        self.arom_exp   = np.zeros(self.L)                  # FIX 1: self.L not L
        self.res_keys   = []

        #---- sasa
        sasa_atom = struc.sasa(self.heavy, 1.4, vdw_radii="ProtOr")
        sasa_atom = np.nan_to_num(sasa_atom, nan=0.0).astype(np.float64)   # explicit float64
        self.sasa_atom = sasa_atom
        
        sc_mask = ~np.isin(self.heavy.atom_name, list(BACKBONE_ATOMS))
        sasa_atom_sc = sasa_atom * sc_mask.astype(float)
        
        starts = struc.get_residue_starts(self.heavy)
        self.sasa_resi = np.add.reduceat(sasa_atom,    starts)   # stays float64
        self.sasa_sch  = np.add.reduceat(sasa_atom_sc, starts)
        #---------
        self.ref_sasa = {k: v * 100.0 for k, v in REF_SASA.items()}

        for i, res in enumerate(self.residues):
            resname = res.res_name[0]
            chain   = res.chain_id[0]
            res_id  = res.res_id[0]
            ins     = res.ins_code[0]
            self.res_keys.append(f"{chain}_{res_id}_{ins}_{resname}")

            ca_mask = res.atom_name == "CA"
            self.ca_coords[i] = (
                res.coord[ca_mask][0] if ca_mask.any()
                else res.coord.mean(axis=0)
            )

            self.charges[i] = RES_CHARGE.get(resname, 0.0)
            self.hydros[i]  = max(HYDRO[0].get(resname, 0.0), 0.0)  # clip negatives
            self.hydro_full[i]  = max(HYDRO[1].get(resname, 0.0), 0.0)  # clip negatives

            ref  = self.ref_sasa.get(resname)
            sasa = self.sasa_resi[i]
            sasa_part = self.sasa_sch[i]
            if ref is not None and ref > 0:
                self.exposures[i] = float(np.clip(sasa / ref, 0.0, 1.0))
                self.exposures_sch[i] = float(np.clip(sasa_part / ref, 0.0, 1.0))
            
            self.hydro_exp[i] = self.hydros[i] * self.exposures_sch[i]

            self.charge_exp[i] = (
                self.charges[i] if self.exposures[i] >  0.25 else 0.0
            )

            # aromatic exposure — zero for non-aromatic or missing ref
            if ref is None or ref <= 0 or resname not in AROMATIC:
                self.arom_exp[i] = 0.0
            else:
                self.arom_exp[i] = float(np.clip(sasa_part, 0.0, 1.0))             
       
#----------------------------------------------------


class MaskedFeatureNormalizer:
    """
    Per-feature normalization for X with shape (M, L, N),
    using a residue mask with shape (M, L).

    Normalizes each feature n across all valid residues in the training set.
    """

    def __init__(self, eps=1e-8, clip=None):
        self.eps = eps
        self.clip = clip
        self.mean_ = None
        self.std_ = None

    def fit(self, X, mask):
        X = np.asarray(X, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)

        if X.ndim != 3:
            raise ValueError(f"X must have shape (M, L, N), got {X.shape}")
        if mask.ndim != 2:
            raise ValueError(f"mask must have shape (M, L), got {mask.shape}")
        if X.shape[:2] != mask.shape:
            raise ValueError(f"X shape {X.shape[:2]} and mask shape {mask.shape} do not match")

        # Select only valid residues: result shape = (num_valid_residues, N)
        X_valid = X[mask]

        if X_valid.shape[0] == 0:
            raise ValueError("Mask contains no valid residues")

        self.mean_ = X_valid.mean(axis=0)
        self.std_ = X_valid.std(axis=0)

        # Avoid divide-by-zero for constant features
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)

        return self

    def transform(self, X, mask, zero_pad=True):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Call fit() before transform()")

        X = np.asarray(X, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)

        if X.ndim != 3:
            raise ValueError(f"X must have shape (M, L, N), got {X.shape}")
        if mask.ndim != 2:
            raise ValueError(f"mask must have shape (M, L), got {mask.shape}")
        if X.shape[:2] != mask.shape:
            raise ValueError(f"X shape {X.shape[:2]} and mask shape {mask.shape} do not match")
        if X.shape[2] != self.mean_.shape[0]:
            raise ValueError(
                f"Feature count mismatch: X has {X.shape[2]}, normalizer has {self.mean_.shape[0]}"
            )

        Xn = (X - self.mean_[None, None, :]) / self.std_[None, None, :]

        if self.clip is not None:
            Xn = np.clip(Xn, -self.clip, self.clip)

        if zero_pad:
            Xn[~mask] = 0.0

        return Xn

    def fit_transform(self, X, mask, zero_pad=True):
        return self.fit(X, mask).transform(X, mask, zero_pad=zero_pad)

#---------------------------------------------------
def normalize_train_val_test(X_train, mask_train, X_val=None, mask_val=None, X_test=None, mask_test=None,
                             clip=None):
    """
    Fit on training only, then apply to train/val/test.
    """
    norm = MaskedFeatureNormalizer(clip=clip)
    norm.fit(X_train, mask_train)

    out = {
        "X_train": norm.transform(X_train, mask_train),
        "mean": norm.mean_,
        "std": norm.std_,
        "normalizer": norm,
    }

    if X_val is not None:
        if mask_val is None:
            raise ValueError("mask_val is required when X_val is provided")
        out["X_val"] = norm.transform(X_val, mask_val)

    if X_test is not None:
        if mask_test is None:
            raise ValueError("mask_test is required when X_test is provided")
        out["X_test"] = norm.transform(X_test, mask_test)

    return out
#-------------------------------------------

def extract_features_batch(pdb_paths: list,
                           #label_list: list,
                            Ab: list,
                            vh: list,
                            vl: list,
                            source_list: list,
                            DIR_DATA_PSR: str, # location of Hidrophobisicy_score.xlsx and Residue_max_SASA.xlsx
                            Score = "WiWh",
                            max_len: int = 290,
                            k_neighbors: int = 10,
                           contact_thresholds = {"large": [6.0, 9.0, 12.0, 16.0], "small": [6.0, 9.0, 12.0], "selected": [0,2]}
 
                          ) -> dict:
    print("**************   start ************")
    Score_table = os.path.join(DIR_DATA_PSR,'Hidrophobisicy_score.xlsx')
    SASA_table = os.path.join(DIR_DATA_PSR,'Residue_max_SASA.xlsx')
    
    REF_SASA = Read_SASA_Free_Residue(SASA_table)
    HYDRO_POS = [Read_Hy_Score(Score_table, Score = Score, Norm = ""),  Read_Hy_Score(Score_table, Score = "Ja", Norm = "TAP")]

    
    # ── Loading back ──────────────────────────────────────────────────
    # data = np.load("struct_features.npz", allow_pickle=True)
    # combined = data["combined"]   # [B, L, 17] if k=10
    #
    # ── In PyTorch DataLoader ─────────────────────────────────────────
    # import torch
    # struct_feat = torch.from_numpy(combined)  # [B, L, 17]
    # # Pass to model alongside AbLang embeddings:
    # logits = model(ablang_emb, struct_feat)
    
    batch = {
        "angles":              [],
        "sasa":                [],
        "sasa_sch":                [],
        "neighbors":           [],
        "hydro_sum":               [],
        "charge_sum":              [],
        "h_bonds":             [],
        "contacts":            [],
        "contacts_interchain": [],
        "aromatic_sum":            [],
        "q_dipole":            [],
        "potential_sites":     [],
         "hydro_patch_sizes":  [],
        "cdr1_H":              [],
        "cdr2_H":              [],
        "cdr3_H":              [],
        "cdr1_L":              [],
        "cdr2_L":              [],
        "cdr3_L":              [],
        "charge_cdrs":         [],
        "charge_exp":             [],
        "ss_num":              [],
        "cdr_fw":              [],
        "q_dipole_h_min":     [],
        "q_dipole_h_abs_diff": [],
        "q_dipole_h_product":  [],
        "q_dipole_h_sum":      [],
        "mask":                [],
        "pdb_paths":           [],
        "vh":                  [],
        "vl":                  [],
        "IMGT_vh":                  [],
        "IMGT_vl":                  [],
        "source":              [],      
        "CA":                  [],
        "Ab_ID":               [],
        "B2A":                  [],      
        "A2B":                  [],
        "flatness_score_hy":    [],
        "flatness_score":       [],
        "salt_bridge_bonded_score":       [],
        "salt_bridge_unused_score":       [],
        "pos_potential":       [],
        "neg_potential":       [],
        "hydro_full_cdr":      [],
        #"charge_chain":      [],
        "charged_patch_area_pos": [],
        "charged_patch_area_neg": [],
        "pI_and_charge" : []
    }

    # Declared before the loop so the failure branch can tell a metadata key from a
    # feature key; the same sets are used again when stacking, below.
    OBJECT_KEYS_ = {"pdb_paths", "Ab_ID", "source", "IMGT_vh", "IMGT_vl"}
    RAGGED_KEYS_ = {"B2A", "A2B"}
    failed_idx = []

    #for i, (pdb_path,label,VH,VL, Source,Ab_ID) in enumerate(zip(pdb_paths,label_list,vh,vl,source_list,Ab)):
    for i, (pdb_path,VH,VL, Source,Ab_ID) in enumerate(zip(pdb_paths,vh,vl,source_list,Ab)):
        #if i <501: continue
        if i%100 == 0 : 
            print(f"[{i+1}/{len(pdb_paths)}] Processing {os.path.basename(pdb_path)}")
        meta_map = {
                "pdb_paths": pdb_path,
                #"labels": label,
                "vh": VH,
                "vl": VL,
                "Ab_ID": Ab_ID,
                "source": Source  ,
            }

        try:
            feats = extract_features_single(pdb_path, 
                                            VH,
                                            VL,
                                            REF_SASA,
                                            HYDRO_POS,
                                            max_len, 
                                            k_neighbors,
                                            contact_thresholds,
                                           )
            for key in batch:
                if key in meta_map:
                    batch[key].append(meta_map[key])
                else:
                    try:
                        batch[key].append(feats[key])
                    except:
                        print("!!!!!!!!!!!!!! ERROR !!!!!!!!!!!!!!!!!!  ",key,feats[key].shape)
            
        except Exception as e:
            print(f"  WARNING: failed on {pdb_path}: {e}")
            failed_idx.append(i)
            # Placeholder of the RIGHT SHAPE for each key.
            #
            # Appending np.zeros((max_len, 1)) for every key -- as this did -- gives
            # "mask" shape (max_len, 1) where every other row has (max_len,), and
            # "pI_and_charge" (max_len, 1) where every other row has (7,). The
            # np.stack below then raises "all input arrays must have the same
            # shape", so ONE unreadable structure killed the whole batch. Shapes
            # are taken from the first successful molecule; when the failure comes
            # first they are filled in afterwards.
            for key in batch:
                if key in meta_map:
                    batch[key].append(meta_map[key])
                elif key in OBJECT_KEYS_ or key in RAGGED_KEYS_:
                    batch[key].append([])
                elif batch[key]:
                    batch[key].append(np.zeros_like(np.asarray(batch[key][0],
                                                               dtype=float)))
                else:
                    batch[key].append(None)          # patched below

    # Failures before the first success have no shape to copy yet.
    for key, vals in batch.items():
        if key in OBJECT_KEYS_ or key in RAGGED_KEYS_:
            continue
        ref = next((v for v in vals if v is not None), None)
        if ref is None:
            continue
        for j, v in enumerate(vals):
            if v is None:
                vals[j] = np.zeros_like(np.asarray(ref, dtype=float))
    if failed_idx:
        print(f"  {len(failed_idx)} structure(s) failed and are zero-filled: "
              f"rows {failed_idx[:10]}{'...' if len(failed_idx) > 10 else ''}. "
              "Their mask is all-zero, so drop them before training.")
    print("END")
   # return {k: np.stack(v, axis=0) for k, v in batch.items()}
    OBJECT_KEYS, RAGGED_KEYS = OBJECT_KEYS_, RAGGED_KEYS_

    result = {}
    for k, v in batch.items():
        if k in OBJECT_KEYS or k in RAGGED_KEYS:
            arr = np.empty(len(v), dtype=object)
            arr[:] = v                       # never let numpy broadcast ragged lists
            result[k] = arr
        else:
            shapes = {np.asarray(x).shape for x in v}
            if len(shapes) > 1:
                raise ValueError(
                    f"feature '{k}' has inconsistent shapes across molecules: "
                    f"{sorted(shapes)}. This is a bug in extract_features_single "
                    "or in the failure placeholder, not bad input.")
            result[k] = np.stack(v, axis=0)
    
    return result

#-----------------------------------------------------------------------------------------
mapping_sss_full = {
    'H': 0,  # alpha helix
    'G': 1,  # 3-10 helix
    'I': 2,  # pi helix
    'E': 3,  # strand
    'B': 4,  # bridge
    'T': 5,  # turn
    'S': 6,  # bend
    ' ': 7   # loop / undefined
}    
def extract_features_single(pdb_path: str,
                            vh: str,
                            vl: str,
                            REF_SASA: str,
                            HYDRO_POS: list,
                            max_len: int = 290,
                            k_neighbors: int = 6 ,
                            contact_thresholds = {"large": [6.0, 9.0, 12.0, 16.0], "small": [6.0, 9.0, 12.0], "selected": [0,2]}
                           ) -> dict:  
     
    structure = load_structure(pdb_path)

    IMGT_vh=get_indices(structure, scheme="imgt", chain_type="H", keep_insertions=True)
    IMGT_vl=get_indices(structure, scheme="imgt", chain_type="L", keep_insertions=True)
    angles    = get_backbone_dihedrals(structure)          # [L, 3]

    Ab = ab_structure(structure,
                      REF_SASA,
                      HYDRO_POS)  
    if Ab.protonated is False: print(pdb_path)
    sasa_total = Ab.sasa_resi[:,None]
    sasa_sch = Ab.sasa_sch[:,None]
    CA = Ab.ca_coords  # [L, 3]
    L_actual  = len(Ab.ca_coords)
    neighbors = get_residue_neighbors(structure, k=k_neighbors)                      # [L, k]      For each Cα, distances to k nearest Cα neighbors (excluding self and sequential n±1)
    _, h_bonds  = count_hbonds_per_residue(structure)                           # [L, 1]      number of h-bonds per residue 


    contacts,contacts_interchain  =  get_contact_features(structure,thresholds=contact_thresholds["small"]) # 2x[L, 3] number of contact residues per residue + same for interchain
    hydro_sum, charge_sum, aromatic_sum  = aro_hydro_charge(Ab,contact_thresholds["small"])   # [L,1]+[L,2] number of hydrophobic, aromatic , and charge contacts
    q_dipole = get_quasi_dipole(Ab, exposure_threshold=0.1, thresholds=contact_thresholds["large"])    # [L, 3] quasy-dipol per residue (q1-q2)*d , whre q1 and q2 symmetric exposed charges
    potential_sites = get_surface_potential_sites(Ab,scheme="imgt")    # [L,1] 
    _, hydro_patch_sizes, _ = get_hydrophobic_patches(Ab, hydro_threshold=0.25,  distance_cutoff=10.0,  exposure_threshold=0.1)

    q_dipole_h = get_hydrophobic_dipoles(Ab, exposure_threshold=0.1, thresholds=contact_thresholds["small"])
    q_dipole_h_min , q_dipole_h_abs_diff, q_dipole_h_product, q_dipole_h_sum = q_dipole_h["min"], q_dipole_h["abs_diff"], q_dipole_h["product"], q_dipole_h["sum"]


    cdr1_H ,charge_cdr1_H = get_cdr_flags(Ab,  cdr=1, chain="H")      # [L, 1]
    cdr2_H ,charge_cdr2_H= get_cdr_flags(Ab,  cdr=2, chain="H")      # [L, 1]
    cdr3_H ,charge_cdr3_H = get_cdr_flags(Ab,  cdr=3, chain="H")      # [L, 1]
    cdr1_L ,charge_cdr1_L = get_cdr_flags(Ab,  cdr=1, chain="L")      # [L, 1]
    cdr2_L ,charge_cdr2_L = get_cdr_flags(Ab,  cdr=2, chain="L")      # [L, 1]
    cdr3_L ,charge_cdr3_L = get_cdr_flags(Ab,  cdr=3, chain="L")      # [L, 1]    # Build mask before padding

    charge_cdrs = cdr1_H*charge_cdr1_H +cdr2_H*charge_cdr2_H+cdr3_H*charge_cdr3_H+cdr1_L*charge_cdr1_L +cdr2_L*charge_cdr2_L+cdr3_L*charge_cdr3_L
    cdrs = cdr1_H.flatten()+cdr2_H.flatten()+cdr3_H.flatten()+cdr1_L.flatten()+cdr2_L.flatten()+cdr3_L.flatten()
    pos_potential, neg_potential = compute_residue_potentials(Ab,probe_distance=[4.0, 16.0],use_sidechain_exposure= True)
    hydro_full_cdr = (Ab.hydro_full *cdrs)[:, None]

    charged_patch = compute_charged_patch_features(Ab, radii=contact_thresholds["small"])
    
    #---------- remove highly correlated features
    charge_sum = charge_sum[...,contact_thresholds["selected"]]
    aromatic_sum =aromatic_sum[...,contact_thresholds["selected"]]
    hydro_sum = hydro_sum[...,contact_thresholds["selected"]]
    contacts_interchain  = contacts_interchain[...,contact_thresholds["selected"]]
    charged_patch_area_pos = charged_patch["pos_area"][...,contact_thresholds["selected"]]
    charged_patch_area_neg = charged_patch["neg_area"][...,contact_thresholds["selected"]]
    #----------------------
    
    #cys_bonds = cys_bridges(Ab)

    traj = md.load(pdb_path)
    encode = np.vectorize(lambda x: mapping_sss_full.get(x, -1))
    ss = md.compute_dssp(traj, simplified=True) 
    ss_num = encode(ss).T
    
    cdr_fw = cdr_fw_contact_counts(Ab, {"cdr1_H":cdr1_H, "cdr2_H":cdr2_H, "cdr3_H":cdr3_H, "cdr1_L":cdr1_L, "cdr2_L":cdr2_L, "cdr3_L":cdr3_L}, polar_cutoff=3.5, count="both")
    biotite_to_ablang, ablang_to_biotite = build_ablang_index_map(Ab, vh, vl, has_separator=False)
    patch_flat_score = hydrophobic_flat_score(Ab, radius=8.0, sigma=1.5)
    flatness_score= patch_flat_score["patch_flat"][:, None]
    flatness_score_hy= patch_flat_score["patch_hydro"][:, None]
    ionic_contacts = compute_ionic_contacts_split(Ab)
    bonded_score = ionic_contacts["bonded_score"][:, None]
    unused_score = ionic_contacts["unused_score"][:, None]

    propka_out = fv_pi(pdb_path, ph_list=(7.5, 6.0))
    pI_and_charge = [ propka_out["pI_folded"],  propka_out["pI_unfolded"],  propka_out["net_charge_ph7.5"],propka_out["net_charge_ph6.0"]]
    is_lambda = get_light_isotype(Ab)   
    pI_and_charge = np.append(np.ravel(pI_and_charge), np.float32(is_lambda))
    cH, cL = get_chain_charges(Ab)
    pI_and_charge = np.append(np.ravel(pI_and_charge), [cH, cL])
    
    #charge_chain = per_residue_chain_charge(Ab)[:, None] 

    charge_exp = Ab.charge_exp[:, None]
    mask = np.zeros(max_len, dtype=np.float32)
    mask[:min(L_actual, max_len)] = 1.0

    
    
    return {
        "angles":          pad_or_trim(angles,    max_len),
        "sasa":            pad_or_trim(sasa_total, max_len),
        "sasa_sch":        pad_or_trim(sasa_sch, max_len),
        "neighbors":       pad_or_trim(neighbors, max_len),
        "hydro_sum":           pad_or_trim(hydro_sum, max_len),
        "charge_sum":          pad_or_trim(charge_sum, max_len),
        "h_bonds":         pad_or_trim(h_bonds, max_len),
        "contacts":        pad_or_trim(contacts, max_len),
        "contacts_interchain":   pad_or_trim(contacts_interchain, max_len),
        "aromatic_sum":        pad_or_trim(aromatic_sum, max_len),
        "q_dipole":        pad_or_trim(q_dipole, max_len),
        "potential_sites": pad_or_trim(potential_sites, max_len),
        "hydro_patch_sizes": pad_or_trim(hydro_patch_sizes, max_len),
        "cdr1_H":          pad_or_trim(cdr1_H, max_len),
        "cdr2_H":          pad_or_trim(cdr2_H, max_len),
        "cdr3_H":          pad_or_trim(cdr3_H, max_len),
        "cdr1_L":          pad_or_trim(cdr1_L, max_len),
        "cdr2_L":          pad_or_trim(cdr2_L, max_len),
        "cdr3_L":          pad_or_trim(cdr3_L, max_len),
        "cdr3_L":          pad_or_trim(cdr3_L, max_len),
        "charge_cdrs":     pad_or_trim(charge_cdrs, max_len),
        "ss_num":          pad_or_trim(ss_num, max_len),
        "cdr_fw":          pad_or_trim(cdr_fw, max_len),
        "q_dipole_h_min":  pad_or_trim(q_dipole_h_min, max_len),
        "q_dipole_h_abs_diff":   pad_or_trim(q_dipole_h_abs_diff, max_len),
        "q_dipole_h_product":   pad_or_trim(q_dipole_h_product, max_len),
        "q_dipole_h_sum":   pad_or_trim(q_dipole_h_sum, max_len),
        "CA":             pad_or_trim(CA, max_len),
        "mask":       mask,
        "IMGT_vh":    IMGT_vh,
        "IMGT_vl":    IMGT_vl,
        "B2A":        biotite_to_ablang, 
        "A2B":        ablang_to_biotite,
        "flatness_score_hy":  pad_or_trim(flatness_score_hy, max_len),
        "flatness_score":  pad_or_trim(flatness_score, max_len),
        "salt_bridge_bonded_score":  pad_or_trim(bonded_score, max_len),
        "salt_bridge_unused_score":  pad_or_trim(unused_score, max_len),
        "pos_potential": pad_or_trim(pos_potential, max_len),
        "neg_potential": pad_or_trim(neg_potential, max_len),
        "hydro_full_cdr": pad_or_trim(hydro_full_cdr, max_len),
        #"charge_chain": pad_or_trim(charge_chain, max_len),
        "charge_exp": pad_or_trim(charge_exp, max_len),
        "charged_patch_area_pos": pad_or_trim(charged_patch_area_pos, max_len),
        "charged_patch_area_neg": pad_or_trim(charged_patch_area_neg, max_len),
        "pI_and_charge": pI_and_charge
    }

#-----------------------------------------------------------------------------------------

def combine_features(feats: dict) -> np.ndarray:
    """
    Concatenate all features into a single tensor.
    angles(3) + sasa(1) + ss(3) + neighbors(k) = 7+k channels

    Returns [B, max_len, 7+k]
    """
    return np.concatenate([
        feats["angles"],     # [B, L, 3]  0...2
        feats["sasa"],       # [B, L, 1]  3
        feats["sasa_sch"],       # [B, L, 1]  3
        feats["neighbors"],  # [B, L, 10]  4...14
        feats["hydro_sum"],     # [B, L, 4] 15...17....20   
        feats["charge_sum"],     # [B, L, 4] 15...17....20   
        feats["h_bonds"],     # [B, L, 1]    21 
        feats["contacts"],     # [B, L, 3]   22...24    
        feats["contacts_interchain"],     # [B, L, 3]  25...27
        feats["aromatic_sum"],     # [B, L, 3]   28... 30
        feats["q_dipole"],     # [B, L, 3]  31...33
        feats["potential_sites"],     # [B, L, 1]  34...36
        feats["hydro_patch_sizes"],  # [B, L, 1]   37
        feats["cdr1_H"],     # [B, L, 1]    38
        feats["cdr2_H"],     # [B, L, 1]   39
        feats["cdr3_H"],     # [B, L, 1]   40
        feats["cdr1_L"],     # [B, L, 1]   41
        feats["cdr2_L"],     # [B, L, 1]   42
        feats["cdr3_L"],     # [B, L, 1]   423
        feats["charge_cdrs"],
        feats["cdr_fw"],
        feats["ss_num"],
        feats["q_dipole_h_min"],
        feats["q_dipole_h_abs_diff"],
        feats["q_dipole_h_product"],
        feats["q_dipole_h_sum"],
        feats["flatness_score_hy"],
        feats["flatness_score"],
        feats["salt_bridge_bonded_score"],
        feats["salt_bridge_unused_score"],
        feats["pos_potential"],
        feats["neg_potential"],
        feats["hydro_full_cdr"],
        #feats["charge_chain"],    
        feats["charge_exp"] ,
        feats["charged_patch_area_pos"],
        feats["charged_patch_area_neg"]
    ], axis=-1)  
#-----------------------------------------------------------------------------------------  
    
def load_structure(pdb_path: str) -> struc.AtomArray:
    """Load first model from a PDB file as an AtomArray."""
    f = pdb.PDBFile.read(pdb_path)
    structure = pdb.get_structure(f, model=1)
    # Keep only amino acids (drop HETATM, water, etc.)
    structure = structure[struc.filter_amino_acids(structure)]
    return structure
#-----------------------------------------------------------------------------------------

def get_backbone_dihedrals(structure: struc.AtomArray) -> np.ndarray:
    """
    Compute φ, ψ, ω per residue.
    Returns [L, 3] in radians. NaN for terminal residues.
    """
    phi,  psi,  omega = struc.dihedral_backbone(structure)
    # phi/psi/omega are [L] arrays; stack → [L, 3]
    angles = np.stack([phi, psi, omega], axis=-1)   # [L, 3]
    # Replace NaN with 0.0 (terminals / missing atoms)
    angles = np.nan_to_num(angles, nan=0.0)
    return angles.astype(np.float32)
   
#-----------------------------------------------------------------------------------------

from scipy.spatial.distance import cdist

def get_ca_neighbors(structure: struc.AtomArray, k: int = 10) -> np.ndarray:
    """
    For each residue i, find the k nearest nonlocal residues using:
        d(i, j) = min distance from CA of residue i
                  to any heavy atom in residue j
    Excludes residue i itself and its immediate sequence neighbors.
    
    Returns shape (L, k), zero-padded if fewer than k valid neighbors.
    """
    residue_starts = struc.get_residue_starts(structure)
    residue_stops  = np.concatenate([residue_starts[1:], [len(structure)]])
    L = len(residue_starts)

    # Build CA coords and a flat heavy-atom array with per-atom residue index
    ca_coords      = np.full((L, 3), np.nan, dtype=np.float64)
    heavy_chunks   = []
    res_idx_chunks = []

    has_element = (hasattr(structure, "element")
                   and structure.element is not None
                   and len(structure.element) > 0)

    for i, (start, stop) in enumerate(zip(residue_starts, residue_stops)):
        res_atoms = structure[start:stop]
        ca_mask = (res_atoms.atom_name == "CA")
        if ca_mask.any():
            ca_coords[i] = res_atoms.coord[ca_mask][0]

        if has_element:
            heavy_mask = (res_atoms.element != "H")
        else:
            names = np.asarray(res_atoms.atom_name).astype(str)
            heavy_mask = ~np.char.startswith(names, "H")

        if heavy_mask.any():
            heavy_chunks.append(res_atoms.coord[heavy_mask])
            res_idx_chunks.append(np.full(int(heavy_mask.sum()), i, dtype=np.int32))

    if not heavy_chunks:
        return np.zeros((L, k), dtype=np.float32)

    heavy_flat = np.concatenate(heavy_chunks, axis=0)   # (N_heavy, 3)
    atom_res   = np.concatenate(res_idx_chunks)         # (N_heavy,)

    # Mask CAs with NaN for special handling at the end
    nan_ca = np.isnan(ca_coords).any(axis=1)
    ca_safe = np.where(nan_ca[:, None], 0.0, ca_coords)

    # All CA-to-heavy distances in one shot: (L, N_heavy)
    all_dists = cdist(ca_safe, heavy_flat)

    # Group atoms by residue: min distance per (i, j) pair
    sort_order   = np.argsort(atom_res, kind="stable")
    sorted_dists = all_dists[:, sort_order]
    sorted_res   = atom_res[sort_order]
    unique_res, group_starts = np.unique(sorted_res, return_index=True)
    grouped = np.minimum.reduceat(sorted_dists, group_starts, axis=1)   # (L, n_unique)

    # Project back to (L, L), filling missing residues with +inf
    res_dists = np.full((L, L), np.inf, dtype=np.float64)
    res_dists[:, unique_res] = grouped

    # Exclude self and immediate sequence neighbors
    diag = np.arange(L)
    res_dists[diag, diag] = np.inf
    if L > 1:
        res_dists[diag[:-1], diag[1:]] = np.inf
        res_dists[diag[1:],  diag[:-1]] = np.inf

    # Top-k smallest per row, ascending
    top_k = np.partition(res_dists, kth=min(k, L - 1), axis=1)[:, :k]
    top_k = np.sort(top_k, axis=1)

    # Replace inf (missing neighbors) and NaN-CA rows with 0
    top_k = np.where(np.isinf(top_k), 0.0, top_k)
    top_k[nan_ca] = 0.0

    return top_k.astype(np.float32)


#-------------
def get_residue_neighbors(structure: struc.AtomArray, k: int = 10) -> np.ndarray:
    """
    For each residue i, return distances to its k nearest non-sequential
    neighbors, where d(i, j) = min distance between any heavy atom of i
    and any heavy atom of j. Symmetric metric.
    """
    residue_starts = struc.get_residue_starts(structure)
    residue_stops  = np.concatenate([residue_starts[1:], [len(structure)]])
    L = len(residue_starts)

    # Flat heavy-atom array with residue index
    has_element = (hasattr(structure, "element")
                   and structure.element is not None
                   and len(structure.element) > 0)
    heavy_chunks, res_idx_chunks = [], []
    for i, (start, stop) in enumerate(zip(residue_starts, residue_stops)):
        res = structure[start:stop]
        if has_element:
            heavy_mask = (res.element != "H")
        else:
            heavy_mask = ~np.char.startswith(
                np.asarray(res.atom_name).astype(str), "H"
            )
        if heavy_mask.any():
            heavy_chunks.append(res.coord[heavy_mask])
            res_idx_chunks.append(np.full(int(heavy_mask.sum()), i, dtype=np.int32))

    if not heavy_chunks:
        return np.zeros((L, k), dtype=np.float32)

    heavy_flat = np.concatenate(heavy_chunks, axis=0)         # (N, 3)
    atom_res   = np.concatenate(res_idx_chunks)               # (N,)

    # All heavy-heavy distances: (N, N). For typical antibody (~1500 atoms),
    # this is ~9 MB — fine. For huge proteins, switch to cKDTree-based
    # block processing.
    D = cdist(heavy_flat, heavy_flat)

    # Reduce to (L, L): min over (a in i, b in j) of D[a, b]
    # Equivalent to grouping rows then columns by residue
    sort_order = np.argsort(atom_res, kind="stable")
    D = D[sort_order][:, sort_order]
    sorted_res = atom_res[sort_order]
    _, group_starts = np.unique(sorted_res, return_index=True)

    # First reduce along rows (atoms of i) → (L, N)
    row_min = np.minimum.reduceat(D, group_starts, axis=0)
    # Then reduce along columns (atoms of j) → (L, L)
    res_dists = np.minimum.reduceat(row_min, group_starts, axis=1)

    # Exclude self and sequence-adjacent neighbors
    diag = np.arange(L)
    res_dists[diag, diag] = np.inf
    if L > 1:
        res_dists[diag[:-1], diag[1:]] = np.inf
        res_dists[diag[1:],  diag[:-1]] = np.inf

    # Top-k smallest per row, sorted ascending
    k_eff = min(k, L - 1)
    top_k = np.partition(res_dists, kth=k_eff, axis=1)[:, :k]
    top_k = np.sort(top_k, axis=1)
    top_k = np.where(np.isinf(top_k), 0.0, top_k).astype(np.float32)
    return top_k.astype(np.float32)
#----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------

def pad_or_trim(arr: np.ndarray, max_len: int, pad_value=np.nan) -> np.ndarray:
    """Pad (with pad_value) or trim first axis to max_len."""
    L = arr.shape[0]
    if L >= max_len:
        return arr[:max_len]
    """
    pad_shape = (max_len - L,) + arr.shape[1:]
    pad_block = np.full(pad_shape, pad_value, dtype=arr.dtype)
    return np.concatenate([arr, pad_block], axis=0)
    """
    dt = arr.dtype
    if np.issubdtype(dt, np.integer) and not np.isfinite(pad_value):
        arr = arr.astype(np.float32); dt = arr.dtype
    pad_block = np.full((max_len - L,) + arr.shape[1:], pad_value, dtype=dt)
    return np.concatenate([arr, pad_block], axis=0)
    
#-----------------------------------------------------------------------------------------

def Read_SASA_Free_Residue(SASA_table, L = False):
    
    # table with 
    SASA_data = pd.read_excel(SASA_table)
    SASA_dictionary_3L = {}
    SASA_dictionary_1L = {}
    for r3,r1,h in zip(SASA_data.Residue3L.str.upper().values,SASA_data.Residue1L.str.upper().values, SASA_data.Empirical.values):
        SASA_dictionary_3L[r3] = float(h)/100.0
        SASA_dictionary_1L[r1] = float(h)/100.0
    if L:     return(SASA_dictionary_1L)

    return(SASA_dictionary_3L)



#-----------------------------------------------------------------------------------------

def Read_Hy_Score(Score_table, Score = "WiWh", Norm = ""):

    Hy_data = pd.read_excel(Score_table)
    Hy_dictionary_3L = {}
    for r,h in zip(Hy_data.resi.values, Hy_data[Score]):
        if Norm == "TAP":
            # norm between 1 and 2
            Hy_dictionary_3L[r] = (h-Hy_data[Score].min())/(Hy_data[Score].max()-Hy_data[Score].min()) +1
        elif Norm == "Park":
            # norm between -1 and 2
            Hy_dictionary_3L[r] = 10*(h-Hy_data[Score].min())/(Hy_data[Score].max()-Hy_data[Score].min()) -5
        else:
            Hy_dictionary_3L[r]  = h
    return(Hy_dictionary_3L)

#-----------------------------------
# Hydrogen relaxation budget, in seconds per molecule, and the iteration cap.
# `add_hydrogen` places hydrogens from ideal geometry and is quick; it is
# `relax_hydrogen` that optimises every rotatable OH / NH3+ / SH orientation,
# iteratively and -- by default -- without a limit. On an occasional structure
# with a bad contact from the builder it does not visibly terminate, and a batch
# of 800 molecules then stalls on one of them with no indication which.
#
# Both are module-level so a caller can change them once for a whole run:
#     import features3d as F; F.RELAX_TIMEOUT_S = 0   # skip relaxation entirely
# A panel must be built with ONE setting: relaxed and unrelaxed hydrogens give
# different H-bond counts, and mixing them puts two different quantities into
# the same feature column.
RELAX_TIMEOUT_S = 30.0      # 0 or None -> do not relax at all
RELAX_ITERATIONS = None     # int -> cap hydride's own iteration count


def _relax_with_budget(protonated):
    """`relax_hydrogen` under a wall-clock budget, on a worker thread.

    A thread rather than a process because the AtomArray would have to be
    pickled either way and hydride releases no lock we can interrupt. The thread
    is left to finish in the background if it overruns -- it cannot be killed,
    but it is harmless: nothing reads its result, and it holds only its own copy.
    """
    import threading

    out = {}

    def _work():
        try:
            out["coord"] = (hydride.relax_hydrogen(protonated, RELAX_ITERATIONS)
                            if RELAX_ITERATIONS else hydride.relax_hydrogen(protonated))
        except TypeError:                     # older hydride: no iteration argument
            try:
                out["coord"] = hydride.relax_hydrogen(protonated)
            except Exception as e:            # noqa: BLE001
                out["err"] = e
        except Exception as e:                # noqa: BLE001
            out["err"] = e

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(RELAX_TIMEOUT_S)
    if t.is_alive():
        print(f"  WARNING: relax_hydrogen exceeded {RELAX_TIMEOUT_S:.0f}s -- "
              "keeping ideal-geometry hydrogens for this molecule. Raise "
              "features3d.RELAX_TIMEOUT_S, or set it to 0 for the whole panel.")
        return None
    if "err" in out:
        print(f"  WARNING: relax_hydrogen failed: {out['err']} - using unrelaxed H")
        return None
    return out.get("coord")


def protonation(atom_array, relax=True):
    """
    Add hydrogens to a biotite AtomArray.
    Assumes input is already protein-only (no water/HETATM).
    """
    # Strip existing H to avoid duplicates
    aa = atom_array[atom_array.element != "H"]

    # Bonds required by hydride
    if aa.bonds is None:
        aa.bonds = struc.connect_via_residue_names(aa)

    # Charge annotation required by hydride
    if "charge" not in aa.get_annotation_categories():
        aa.add_annotation("charge", int)
        aa.charge[:] = 0

    try:
        protonated, heavy_mask = hydride.add_hydrogen(aa)
    except Exception as e:
        print(f"  WARNING: hydride.add_hydrogen failed: {e}\n"
               "  Returning heavy-atom structure only.")
        return aa, np.ones(len(aa), dtype=bool)

    if relax and RELAX_TIMEOUT_S:
        coord = _relax_with_budget(protonated)
        if coord is not None:
            protonated.coord = coord

    return protonated, heavy_mask
#-----------------------------------
def count_hbonds_per_residue(atom_array):
    """
    Count how many hydrogen bonds each residue participates in.
    Returns
    -------
    residue_keys : list of tuple
        [(chain_id, res_id, ins_code, res_name), ...] in residue order
    counts : np.ndarray, [L, 1]
        Number of H-bonds each residue participates in.
    """
    if atom_array.bonds is None:
            atom_array.bonds = struc.connect_via_residue_names(atom_array)    
    # Add hydrogens    # check hydrogens   
    aa_mask    = struc.filter_amino_acids(atom_array)
    total_atoms = len(atom_array[aa_mask] )
    h_atoms     = (atom_array[aa_mask].element == "H").sum()
    if h_atoms < total_atoms/3. :
        atom_array, _ = protonation(atom_array)


    # Find hydrogen bonds
    # Returns (n, 3): donor_atom_idx, hydrogen_atom_idx, acceptor_atom_idx
    triplets = struc.hbond(atom_array)

    # Residue boundaries in original order
    starts = struc.get_residue_starts(atom_array)
    stops = np.append(starts[1:], atom_array.array_length())
    n_res = len(starts)

    residue_keys = []
    atom_to_res = np.empty(atom_array.array_length(), dtype=int)

    for res_idx, (start, stop) in enumerate(zip(starts, stops)):
        atom_to_res[start:stop] = res_idx
        residue_keys.append(
            (
                atom_array.chain_id[start],
                int(atom_array.res_id[start]),
                atom_array.ins_code[start],
                atom_array.res_name[start],
            )
        )

    counts = np.zeros(n_res, dtype=np.int32)

    # Count each hydrogen bond once for donor residue and once for acceptor residue
    for donor_i, _, acceptor_i in triplets:
        donor_res = atom_to_res[donor_i]
        acceptor_res = atom_to_res[acceptor_i]

        counts[donor_res] += 1
        if acceptor_res != donor_res:
            counts[acceptor_res] += 1

    return residue_keys, counts[:, None]
    
#----------------------------------
# IMGT CDR positions — works if structure is IMGT numbered
IMGT_CDR_chain = set(range(27,39))  | set(range(56,66))  | set(range(105,118))
IMGT_CDR = {
    "H1": range(27, 39), "H2": range(56, 66), "H3": range(105, 118),
    "L1": range(27, 39), "L2": range(56, 66), "L3": range(105, 118),
}

#------------------------------------------

THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E",
    "GLN":"Q","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K",
    "MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W",
    "TYR":"Y","VAL":"V","MSE":"M",
}

def _residue_metadata(Ab):
    """Returns chain_ids, res_ids, one-letter aa for each residue in Ab.heavy in order."""
    chains, res_ids, aa = [], [], []
    for res in Ab.residues:
        chains.append(res.chain_id[0])
        res_ids.append(int(res.res_id[0]))
        aa.append(THREE_TO_ONE.get(res.res_name[0], "X"))
    return np.array(chains), np.array(res_ids), "".join(aa)


def _imgt_per_residue(seq):
    """Run ANARCI on a sequence, return list[int|None] of IMGT numbers per input position.
    None means that position wasn't part of the variable domain.
    """
    out = anarci([("q", seq)], scheme="imgt", output=False)
    numbering, alignment_details, _ = out
    if numbering[0] is None:
        return [None] * len(seq)
    
    # ANARCI returns: list of (numbering_list, start, end) per domain
    domain = numbering[0][0]   # first (only) domain
    numbering_list, start, end = domain  # numbering_list = [((imgt_num, ins_code), aa), ...]
    
    result = [None] * len(seq)
    seq_pos = start
    for (imgt_num, ins_code), aa in numbering_list:
        if aa == "-":      # gap inserted by ANARCI
            continue
        result[seq_pos] = imgt_num
        seq_pos += 1
    return result

#-----------------------

def _numeric_part(label):
    """'111A' -> 111 ; '27' -> 27"""
    return int("".join(c for c in str(label) if c.isdigit()))

import warnings, numpy as np
warnings.resetwarnings()
#warnings.filterwarnings("error", message="invalid value encountered in cast")

def get_cdr_flags(Ab, cdr=1, chain="H", scheme="imgt"):
    if chain not in ("H", "L"):
        raise ValueError(f"chain must be 'H' or 'L', got {chain}")
    if cdr not in (1, 2, 3):
        raise ValueError(f"cdr must be 1, 2, or 3, got {cdr}")

    cdr_range = set(IMGT_CDR[f"{chain}{cdr}"])

    # Extract per-residue metadata in biotite order
    chains, res_ids, full_seq = _residue_metadata(Ab)
    
    # Cache IMGT numbers per chain to avoid running ANARCI multiple times
    if not hasattr(Ab, "_imgt_numbers"):
        Ab._imgt_numbers = _compute_imgt_numbers(Ab, chains, full_seq)
    imgt_numbers = Ab._imgt_numbers   # list of (chain_letter|None, imgt_num|None) per residue

    flags = np.zeros(Ab.L, dtype=float)
    for i, (chain_assigned, num) in enumerate(imgt_numbers):
        if chain_assigned == chain and num in cdr_range:
            flags[i] = 1.0

    charge_cdr = float(np.sum(flags * Ab.charges))
    if flags.sum() == 0:
        ab_id = getattr(Ab, "Ab_ID", getattr(Ab, "name", "?"))
        seen_chains = sorted({c for c, _ in imgt_numbers if c is not None})
        warnings.warn(
            f"CDR {chain}{cdr} EMPTY for Ab={ab_id}: no residues flagged. "
            f"Chains seen by IMGT numbering: {seen_chains}. "
            f"If a light chain is missing, ANARCI labels kappa as 'K' (not 'L') — "
            f"normalize K->L in _compute_imgt_numbers.",
            RuntimeWarning, stacklevel=2)
    return flags[:, None], charge_cdr
#-------------------------

from scipy.spatial.distance import cdist  # at module top, not inside function

def compute_residue_potentials(
    Ab,
    probe_distance=[4.0, 8.0, 12.0],
    epsilon_water=80.0,
    debye_length=10.0,
    source_exposure_weight=False,
    use_sidechain_exposure=False, 
    distance_cutoff=None,
):
    """
    Compute surface electrostatic potential at every residue, evaluated at
    one or more probe points displaced outward from each residue along the
    side-chain tip direction.

    For each residue i and each probe distance p:
        probe_i = tip_i + p * outward_unit_i
        V(probe_i) = sum_j (q_j * w_j) / (eps_w * d_ij) * exp(-d_ij / lambda_D)

    where:
        d_ij        : distance from probe_i to source j's charge centroid
                      (LYS:NZ, ARG:NE+NH1+NH2, ASP:OD1+OD2, GLU:OE1+OE2,
                       HIS:ND1+NE2 if HIS is charged in Ab.charges).
                      CA fallback for residues without charged side chains.
        outward_i   : unit vector CA -> side-chain tip
                      (farthest heavy side-chain atom from CA).
                      Radial fallback for glycines / missing side chains.
        q_j         : per-residue formal charge from Ab.charges.
        w_j         : exposure of source residue j (if source_exposure_weight=True),
                      else 1.0.
        eps_w       : water dielectric (~80).
        lambda_D    : Debye screening length (~10 Å at physiological salt).
        distance_cutoff : optional hard cutoff; pairs farther than this contribute 0.

    Positive and negative source contributions are kept separate so the
    output reflects how much positive vs. negative field a probe at this
    residue's surface would feel.

    NOTE: No CDR masking is applied here — this is a per-residue feature
    over the entire structure. To extract only CDR residues, multiply the
    returned arrays by your CDR flag vector at the call site, or use
    get_cdr_flags() which does this internally.

    Args:
        Ab                       : ab_structure with .heavy, .L, .charges, .exposures
        probe_distance           : scalar or list of probe offsets (Å) beyond side-chain tip
        epsilon_water            : water dielectric constant
        debye_length             : Debye screening length in Å (None to disable)
        source_exposure_weight   : if True, scale each source charge by its exposure
        distance_cutoff          : optional Å cutoff; pairs beyond contribute 0

    Returns:
        pos_potential : (n_res, n_distances) magnitude of potential from
                        positively charged sources. Non-negative.
        neg_potential : (n_res, n_distances) magnitude of potential from
                        negatively charged sources. Non-negative.
    """
    heavy   = Ab.heavy
    n_res   = Ab.L
    n_atoms = len(heavy)
    coord   = heavy.coord
    rname   = heavy.res_name
    aname   = heavy.atom_name

    res_starts  = struc.get_residue_starts(heavy)
    atom_to_res = np.searchsorted(res_starts, np.arange(n_atoms), side='right') - 1

    # ── charged-atom mask (vectorized via dict) ───────────────────────────
    charged_atom_mask = np.zeros(n_atoms, dtype=bool)
    for resname, atoms in {**POSITIVE_ATOMS, **NEGATIVE_ATOMS}.items():
        is_res = (rname == resname)
        if not is_res.any():
            continue
        is_atom = np.isin(aname, np.asarray(atoms))
        charged_atom_mask |= is_res & is_atom

    # ── source charge centroid per residue (bincount, no Python loop) ─────
    charged_to_res = atom_to_res[charged_atom_mask]
    n_chg = np.bincount(charged_to_res, minlength=n_res).astype(float)
    centre = np.zeros((n_res, 3), dtype=float)
    if charged_to_res.size:
        c_coord = coord[charged_atom_mask]
        for d in range(3):
            centre[:, d] = np.bincount(
                charged_to_res, weights=c_coord[:, d], minlength=n_res
            )
        have_chg = n_chg > 0
        centre[have_chg] /= n_chg[have_chg, None]
        centre[~have_chg] = np.nan

    # ── per-residue all-atom centroid (vectorized) ────────────────────────
    all_count = np.bincount(atom_to_res, minlength=n_res).astype(float)
    all_centroid = np.zeros((n_res, 3), dtype=float)
    for d in range(3):
        all_centroid[:, d] = np.bincount(
            atom_to_res, weights=coord[:, d], minlength=n_res
        )
    nz = all_count > 0
    all_centroid[nz] /= all_count[nz, None]

    # ── CA per residue (vectorized) ───────────────────────────────────────
    ca_mask     = (aname == "CA")
    ca_atom_idx = np.where(ca_mask)[0]
    ca_coords   = np.full((n_res, 3), np.nan, dtype=float)
    ca_coords[atom_to_res[ca_atom_idx]] = coord[ca_atom_idx]
    missing_ca = np.any(np.isnan(ca_coords), axis=1)
    ca_coords[missing_ca] = all_centroid[missing_ca]

    # ── fill missing source centre ────────────────────────────────────────
    missing_centre = np.any(np.isnan(centre), axis=1)
    centre[missing_centre] = ca_coords[missing_centre]
    still_missing = np.any(np.isnan(centre), axis=1)
    centre[still_missing] = all_centroid[still_missing]

    # ── side-chain tip = farthest heavy SC atom from CA (vectorized) ──────
    sc_mask     = ~np.isin(aname, np.asarray(BACKBONE_ATOMS))
    sc_atom_idx = np.where(sc_mask)[0]
    tip_dist    = np.full(n_res, -1.0, dtype=float)
    tip_coords  = np.full((n_res, 3), np.nan, dtype=float)
    if sc_atom_idx.size:
        sc_res_idx = atom_to_res[sc_atom_idx]
        sc_dists   = np.linalg.norm(coord[sc_atom_idx] - ca_coords[sc_res_idx], axis=1)
        # Sort descending by distance, then take first occurrence per residue
        order        = np.argsort(-sc_dists, kind='stable')
        sorted_res   = sc_res_idx[order]
        sorted_atoms = sc_atom_idx[order]
        sorted_dist  = sc_dists[order]
        unique_res, first_idx = np.unique(sorted_res, return_index=True)
        tip_dist[unique_res]   = sorted_dist[first_idx]
        tip_coords[unique_res] = coord[sorted_atoms[first_idx]]
    have_tip = tip_dist > 0

    # ── outward unit vector ───────────────────────────────────────────────
    valid_ca       = ~np.any(np.isnan(ca_coords), axis=1)
    protein_centre = ca_coords[valid_ca].mean(axis=0) if valid_ca.any() else np.zeros(3)
    radial_vec     = ca_coords - protein_centre
    radial_norm    = np.linalg.norm(radial_vec, axis=1, keepdims=True)
    radial_unit    = radial_vec / np.maximum(radial_norm, 1e-9)

    tip_vec  = np.where(have_tip[:, None], tip_coords - ca_coords, 0.0)
    tip_norm = np.linalg.norm(tip_vec, axis=1, keepdims=True)
    use_tip  = tip_norm > 1e-3
    tip_unit = np.where(use_tip, tip_vec / np.maximum(tip_norm, 1e-9), radial_unit)

    # ── probe point ───────────────────────────────────────────────────────
    probe_origin = np.where(have_tip[:, None], tip_coords, ca_coords)

    # ── hoist everything that doesn't depend on probe distance ────────────────
    res_charges = np.asarray(Ab.charges, dtype=float)
    #exposures   = np.clip(np.asarray(Ab.exposures, dtype=float), 0.0, 1.0)
    #source_w    = exposures if source_exposure_weight else np.ones(n_res, dtype=float)
    if source_exposure_weight:
        exp_attr = "exposures_sch" if use_sidechain_exposure else "exposures"
        exposures = np.clip(np.asarray(getattr(Ab, exp_attr), dtype=float), 0.0, 1.0)
        source_w  = exposures
    else:
        source_w = np.ones(n_res, dtype=float)
    q_pos = np.where(res_charges > 0,  res_charges, 0.0) * source_w
    q_neg = np.where(res_charges < 0, -res_charges, 0.0) * source_w

    # ── per-distance probe + potentials ───────────────────────────────────────
    probe_distance = np.atleast_1d(probe_distance)
    pos_potential = []
    neg_potential = []
    for p in probe_distance:
        probe = probe_origin + p * tip_unit            # use p, not probe_distance
    
        dist = cdist(probe, centre)
        np.maximum(dist, 1e-3, out=dist)
    
        factor = 1.0 / (epsilon_water * dist)
        if debye_length is not None and debye_length > 0:
            factor *= np.exp(-dist / debye_length)
        if distance_cutoff is not None:
            factor = np.where(dist > distance_cutoff, 0.0, factor)
    
        pos_potential.append(factor @ q_pos)
        neg_potential.append(factor @ q_neg)
    
    # Stack to (n_distances, n_res) or (n_res, n_distances) — your call:
    pos_potential = np.stack(pos_potential, axis=-1)   # shape (n_res, n_distances)
    neg_potential = np.stack(neg_potential, axis=-1)

    return pos_potential, neg_potential    
    
#---------------------
def _compute_imgt_numbers(Ab, chains, full_seq):
    """For each biotite residue, return (chain_letter, imgt_num) or (None, None) if outside Fv."""
    result = [(None, None)] * Ab.L
    
    def anarci_chain_letter(anarci_output):
        """Extract H or L from ANARCI alignment_details."""
        _, alignment_details, _ = anarci_output
        return alignment_details[0][0]["chain_type"]   # "H", "K", or "L"; treat K as L for kappa
    
    for chain_id_in_struct in np.unique(chains):
        mask = chains == chain_id_in_struct
        chain_seq = "".join(c for c, m in zip(full_seq, mask) if m)
        chain_indices = np.where(mask)[0]
        
        # ANARCI on this chain's sequence
        out = anarci([("q", chain_seq)], scheme="imgt", output=False)
        numbering, _, _ = out
        if numbering[0] is None:
            continue   # not antibody (e.g., antigen chain)
        
        domain = numbering[0][0]
        numbering_list, start, end = domain

        raw_type = anarci_chain_letter(out)            # "H", "K", or "L"
        if raw_type in ("K", "L"):
            Ab._light_isotype = raw_type               # <-- writes the isotype
        chain_type = "L" if raw_type == "K" else raw_type
        
        seq_pos = start
        for (imgt_num, ins_code), aa in numbering_list:
            if aa == "-":
                continue
            biotite_idx = chain_indices[seq_pos]
            result[biotite_idx] = (chain_type, imgt_num)
            seq_pos += 1
    
    return result


#--------------

def get_imgt_positions(atom_array, chain_type="H"):
    """Return a list of IMGT position labels (ints + insertion strings)
    aligned to the residues actually present in atom_array's chain."""
    seq = get_sequence_from_atomarray(atom_array, chain_id=chain_type)
    numbering, _ = number(seq, scheme="imgt")

    positions = []          # one entry per residue that was numbered (no gaps)
    for (imgt_num, ins), aa in numbering:
        if aa == "-":
            continue        # ANARCI gap -> residue not present
        positions.append((int(imgt_num), ins.strip()))
    return positions        # e.g. [(1,''), (2,''), ..., (111,''), (111,'A'), (111,'B'), (112,''), ...]
#----------------------
    
def get_indices(atom_array, scheme="imgt", chain_type="H", keep_insertions=True):
    seq = get_sequence_from_atomarray(atom_array, chain_id=chain_type)
    # A single-domain molecule (VHH) has no L chain, so the sequence is empty and
    # ANARCI returns a falsy numbering rather than a list of positions. These
    # labels are carried as metadata only -- nothing in combine_features reads
    # them -- so an absent chain is an empty list, not an error.
    if not seq:
        return []
    numbering, detected_chain_type = number(seq, scheme=scheme)
    if not numbering:
        return []
    labels = []
    for pos, aa in numbering:
        idx, ins = pos
        if aa == "-":
            continue
        label = f"{idx}{ins.strip()}" if ins.strip() else str(idx)
        labels.append(label if keep_insertions else int("".join(filter(str.isdigit, label))))

    return labels
#----------------------------------------
    
def get_sequence_from_atomarray(atom_array, chain_id=None):
    """
    Extract amino acid sequence from a Biotite AtomArray.
    Optionally filter by chain_id.
    """
    if chain_id is not None:
        atom_array = atom_array[atom_array.chain_id == chain_id]
    
    # Get one representative atom per residue (CA atoms)
    ca_atoms = atom_array[atom_array.atom_name == "CA"]
    
    # res_name gives 3-letter codes
    three_letter = ca_atoms.res_name   # numpy array of strings e.g. ["ALA", "GLY", ...]
    
    # Convert 3-letter → 1-letter
    sequence = "".join([three_to_one(aa)  for aa in three_letter])
    
    return sequence
    
#----------------------------------------

def get_contact_features(atom_array, chain_ids=None, thresholds=[4.0,6.0,8.0]):
    """
    Per-residue interchain contact density using ALL heavy atoms.
    chain_ids: list of chains to include e.g. ["H","L"] or None for all chains
    Returns number of INTERCHAIN contacts within each threshold [L_total, len(thresholds)]
    """
    if chain_ids is not None:
        chain_mask = np.isin(atom_array.chain_id, chain_ids)
    else:
        chain_mask = np.ones(len(atom_array), dtype=bool)
    mask = (
        chain_mask &
        struc.filter_amino_acids(atom_array) &
        (atom_array.element != "H")
    )
    heavy = atom_array[mask]
    res_keys = np.array([
        f"{c}_{r}_{ins}" for c, r, ins in zip(heavy.chain_id, heavy.res_id, heavy.ins_code)
    ])        
    # -- for interchain contacts
    atom_chains = heavy.chain_id                            # [N_atoms]
    # Preserve original order (chain order as they appear in structure)
    unique_res_keys = list(dict.fromkeys(res_keys))
    coords = heavy.coord
    
    # Pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist_ch = np.linalg.norm(diff, axis=-1)
    
    # Mask same-residue atom pairs
    same_res_mask       = (res_keys[:, None] == res_keys[None, :])
    
    # build sequential-neighbour key per atom
    res_indices = np.array([unique_res_keys.index(k) for k in res_keys])  # integer index per atom
    seq_diff = np.abs(res_indices[:, None] - res_indices[None, :])
    dist[seq_diff <= 1] = np.inf   # excludes i-1, i, i+1    
    
    # mask same-chain atom pairs → only interchain distances remain
    same_chain_mask       = (atom_chains[:, None] == atom_chains[None, :])
    dist_ch[same_chain_mask] = np.inf

    features = np.zeros((len(unique_res_keys), len(thresholds)))
    features_ch = np.zeros((len(unique_res_keys), len(thresholds)))
    
    for i, res_key in enumerate(unique_res_keys):
        atom_mask = res_keys == res_key
        res_dists = dist[atom_mask, :]                     # [n_res_atoms, N]
        res_dists_ch = dist_ch[atom_mask, :]                     # [n_res_atoms, N]
        for j, t in enumerate(thresholds):
            contact_atoms    = res_dists < t
            contact_res_keys = res_keys[contact_atoms.any(axis=0)]
            features[i, j]   = len(np.unique(contact_res_keys))
            contact_atoms_ch    = res_dists_ch < t
            contact_res_keys_ch = res_keys[contact_atoms_ch.any(axis=0)]
            features_ch[i, j]   = len(np.unique(contact_res_keys_ch))

    return features.astype(float) , features_ch.astype(float)                       # [L_total, len(thresholds)]

#-------------------------------------
def aro_hydro_charge(Ab, thresholds):
    L = Ab.L
    coords = np.asarray(Ab.ca_coords, dtype=float)
    hydro_exp = np.asarray(Ab.hydro_exp, dtype=float)
    charge_exp = np.asarray(Ab.charge_exp, dtype=float)
    arom_exp = np.asarray(Ab.arom_exp, dtype=float)

    dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)

    local_feat1 = np.zeros((L, len(thresholds) ), dtype=float)
    local_feat3 = np.zeros((L, len(thresholds) ), dtype=float)
    local_feat2 = np.zeros((L, len(thresholds)), dtype=float)

    for i in range(L):
        for j, t in enumerate(thresholds):
            mask = dmat[i] <= t

            local_feat1[i, j] = hydro_exp[mask].sum()
            local_feat3[i, j] = np.sum(charge_exp[i] * charge_exp[mask] / (dmat[i, mask] ** 2))
            local_feat2[i, j] = arom_exp[mask].sum()

    return local_feat1, local_feat3, local_feat2
#-------------------------------------------


#-----------------------

def get_hydrophobic_patches(
    ab,
    hydro_threshold=0.25,    # min hydrophobicity index to be considered hydrophobic
    distance_cutoff=8.0,    # Å — CA-CA distance to be in same patch
    exposure_threshold=0.2,
):
    """
    For each exposed hydrophobic residue (hydro_exp > hydro_threshold):
      - find all spatially connected exposed hydrophobic neighbors (CA < distance_cutoff)
      - compute total hydrophobicity of the connected patch
      - assign patch score to each residue in the patch

    Uses ab.hydro_exp    — precomputed sasa * max(HYDRO, 0) per residue
         ab.exposures    — precomputed SASA ratio
         ab.ca_coords    — (L, 3)
         ab.L            — number of residues

    Returns
    -------
    patch_scores : np.ndarray (L,)
        For each residue: total hydrophobicity of its patch (0 if not in a patch)
    patch_sizes  : np.ndarray (L,)
        Number of residues in the patch this residue belongs to
    patch_ids    : np.ndarray (L,)
        Integer patch id per residue (-1 if not in any patch)
    """
    L          = ab.L
    hydro_exp  = ab.hydro_exp        # (L,) sasa * hydro
    exposures  = ab.exposures_sch        # (L,)
    ca_coords  = ab.ca_coords        # (L, 3)

    # --- 1. identify seed residues: exposed + hydrophobic enough ----------
    hydro_mask = (hydro_exp > hydro_threshold) & (exposures > exposure_threshold)
    hydro_idx  = np.where(hydro_mask)[0]

    if len(hydro_idx) == 0:
        return np.zeros(L), np.zeros(L), np.full(L, -1, dtype=int)

    # --- 2. build adjacency among hydrophobic residues --------------------
    # CA-CA distance matrix restricted to hydrophobic residues only
    hydro_coords = ca_coords[hydro_idx]                        # (n_hydro, 3)
    dist_matrix  = np.linalg.norm(
        hydro_coords[:, None] - hydro_coords[None, :], axis=-1
    )                                                          # (n_hydro, n_hydro)
    adjacency = dist_matrix < distance_cutoff                  # bool (n_hydro, n_hydro)
    np.fill_diagonal(adjacency, False)                         # exclude self

    # --- 3. connected components via union-find ---------------------------
    parent = np.arange(len(hydro_idx))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    rows, cols = np.where(adjacency)
    for r, c in zip(rows, cols):
        union(r, c)

    # normalize: map each node to its root
    roots = np.array([find(k) for k in range(len(hydro_idx))])

    # --- 4. compute patch scores ------------------------------------------
    patch_scores = np.zeros(L,  dtype=float)
    patch_sizes  = np.zeros(L,  dtype=float)
    patch_ids    = np.full(L,   -1, dtype=int)

    unique_roots, patch_id_map = np.unique(roots, return_inverse=False), {}
    for pid, root in enumerate(np.unique(roots)):
        members_local = np.where(roots == root)[0]   # indices into hydro_idx
        members_global = hydro_idx[members_local]    # indices into full residue array

        total_hydro = hydro_exp[members_global].sum()
        patch_size  = len(members_global)

        # only count patches with >1 residue (isolated residues are not a patch)
        if patch_size > 1:
            patch_scores[members_global] = total_hydro
            patch_sizes[members_global]  = patch_size
            patch_ids[members_global]    = pid

    return patch_scores[:, None] , patch_sizes[:, None] , patch_ids[:, None] 

#--------------------
def get_quasi_dipole(Ab, exposure_threshold=0.1, thresholds=[6.0, 8.0, 12.0]):
    """
    For each surface residue i (anchor A):
      - axis     = unit vector COM → CA_i
      - anchor B = surface residue whose COM→CA direction is most opposite to `axis`
                   (argmin of cos(angle) between unit vectors)
      - patch A  = surface residues within Euclidean distance `t` of CA_i
      - patch B  = surface residues within Euclidean distance `t` of CA_B
      - q_A, q_B = summed charges in each patch
      - d        = ||centroid_A - centroid_B||   (3D Euclidean)
      - dipole   = |q_A - q_B| * d

    `thresholds` are 3D patch radii in Å.
    """
    com          = Ab.com
    ca_coords    = Ab.ca_coords
    charges      = Ab.charges
    exposures    = Ab.exposures
    res_keys     = Ab.res_keys
    surface_mask = exposures > exposure_threshold
    surf_idx     = np.where(surface_mask)[0]
    T            = len(thresholds)
    L            = len(res_keys)

    surf_coords     = ca_coords[surface_mask]                  # (L_surf, 3)
    surf_charges    = charges[surface_mask]                    # (L_surf,)
    surf_global_idx = np.where(surface_mask)[0]                # (L_surf,)

    # COM→CA vectors and their unit forms (computed once)
    r_vec       = ca_coords - com
    r_norm      = np.linalg.norm(r_vec, axis=-1).clip(min=1e-8)
    r_unit      = r_vec / r_norm[:, None]                      # (L, 3)
    surf_r_unit = r_unit[surface_mask]                         # (L_surf, 3)

    charge_dip = np.zeros((L, T), dtype=float)

    for i in surf_idx:
        axis = r_unit[i]                                       # (3,)

        # cosine of angle between each surface residue's COM-direction and `axis`
        # most negative cosine = surface residue pointing most opposite to i
        cos_angles      = surf_r_unit @ axis                   # (L_surf,)
        antipode_local  = int(np.argmin(cos_angles))
        anchor_A = ca_coords[i]
        anchor_B = surf_coords[antipode_local]

        # 3D Euclidean distances from every surface residue to each anchor
        d_to_A = np.linalg.norm(surf_coords - anchor_A, axis=-1)
        d_to_B = np.linalg.norm(surf_coords - anchor_B, axis=-1)

        # local index of residue i inside the surface arrays (to exclude self)
        self_local = np.where(surf_global_idx == i)[0]
        self_local = self_local[0] if len(self_local) else None

        for j, t in enumerate(thresholds):
            mask_A = d_to_A <= t
            mask_B = d_to_B <= t

            # if patches overlap, assign shared residues to the closer anchor
            overlap = mask_A & mask_B
            if overlap.any():
                ov_idx = np.where(overlap)[0]
                closer_to_B = d_to_B[ov_idx] < d_to_A[ov_idx]
                mask_A[ov_idx[closer_to_B]] = False
                mask_B[ov_idx[~closer_to_B]] = False

            # exclude self from both patches
            if self_local is not None:
                mask_A[self_local] = False
                mask_B[self_local] = False

            q_A = surf_charges[mask_A].sum()
            q_B = surf_charges[mask_B].sum()

            if mask_A.any() and mask_B.any():
                centroid_A = surf_coords[mask_A].mean(axis=0)
                centroid_B = surf_coords[mask_B].mean(axis=0)
                d = np.linalg.norm(centroid_A - centroid_B)
            else:
                d = 0.0

            charge_dip[i, j] = abs(q_A - q_B) * d

    return charge_dip
    
#---------------------


#----------------------
def get_hydrophobic_dipoles(Ab, exposure_threshold=0.1, thresholds=[6.0, 8.0, 12.0]):
    """
    Computes 4 hydrophobic dipole variants in one pass.
    Uses Ab.hydro_exp (hydrophobicity * side-chain SASA) as the per-residue weight.
    
    Returns dict with keys "min", "abs_diff", "product", "sum", each shape [L, T].
    """
    com           = Ab.com
    ca_coords     = Ab.ca_coords
    h_eff         = Ab.hydro_exp                  # already SASA-weighted
    exposures     = Ab.exposures
    res_keys      = Ab.res_keys
    
    surface_mask = exposures > exposure_threshold
    surf_idx     = np.where(surface_mask)[0]
    T            = len(thresholds)
    L            = len(res_keys)
    
    surf_coords     = ca_coords[surface_mask]
    surf_h_eff      = h_eff[surface_mask]
    surf_global_idx = np.where(surface_mask)[0]
    
    r_vec       = ca_coords - com
    r_norm      = np.linalg.norm(r_vec, axis=-1).clip(min=1e-8)
    r_unit      = r_vec / r_norm[:, None]
    surf_r_unit = r_unit[surface_mask]
    
    out = {
        "min":      np.zeros((L, T), dtype=float),
        "abs_diff": np.zeros((L, T), dtype=float),
        "product":  np.zeros((L, T), dtype=float),
        "sum":      np.zeros((L, T), dtype=float),
    }
    
    for i in surf_idx:
        axis = r_unit[i]
        cos_angles = surf_r_unit @ axis
        antipode_local = int(np.argmin(cos_angles))
        
        anchor_A = ca_coords[i]
        anchor_B = surf_coords[antipode_local]
        
        d_to_A = np.linalg.norm(surf_coords - anchor_A, axis=-1)
        d_to_B = np.linalg.norm(surf_coords - anchor_B, axis=-1)
        
        self_local = np.where(surf_global_idx == i)[0]
        self_local = self_local[0] if len(self_local) else None
        
        for j, t in enumerate(thresholds):
            mask_A = d_to_A <= t
            mask_B = d_to_B <= t
            overlap = mask_A & mask_B
            if overlap.any():
                ov_idx = np.where(overlap)[0]
                closer_to_B = d_to_B[ov_idx] < d_to_A[ov_idx]
                mask_A[ov_idx[closer_to_B]] = False
                mask_B[ov_idx[~closer_to_B]] = False
            if self_local is not None:
                mask_A[self_local] = False
                mask_B[self_local] = False
            
            h_A = surf_h_eff[mask_A].sum()
            h_B = surf_h_eff[mask_B].sum()
            
            if mask_A.any() and mask_B.any():
                centroid_A = surf_coords[mask_A].mean(axis=0)
                centroid_B = surf_coords[mask_B].mean(axis=0)
                d = np.linalg.norm(centroid_A - centroid_B)
            else:
                d = 0.0
            
            out["min"][i, j]      = min(h_A, h_B) * d
            out["abs_diff"][i, j] = abs(h_A - h_B) * d
            out["product"][i, j]  = h_A * h_B * d
            out["sum"][i, j]      = (h_A + h_B) * d
    
    return out
#-------------
def get_rmsf_per_resi(ref, traj):
    """
    RMSF of the backbone — vectorised residue grouping.
    """
    bb_mask_ref  = struc.filter_peptide_backbone(ref) & struc.filter_amino_acids(ref)
    bb_mask_traj = struc.filter_peptide_backbone(traj[0]) & struc.filter_amino_acids(traj[0])

    traj_aligned, _ = struc.superimpose(ref[bb_mask_ref], traj[:, bb_mask_traj])

    bb_coords   = traj_aligned.coord                          # (T, N_atoms, 3)
    mean_coords = bb_coords.mean(axis=0)                      # (N_atoms, 3)
    diff        = bb_coords - mean_coords[None, :, :]         # (T, N_atoms, 3)
    atom_rmsf   = np.sqrt(np.mean(np.sum(diff**2, axis=2), axis=0))  # (N_atoms,)

    # FIX: vectorised residue grouping — no per-residue loop
    bb_atoms = ref[bb_mask_ref]

    # integer residue index per atom — same residue = same integer
    _, res_indices = np.unique(
        np.stack([
            bb_atoms.chain_id,
            bb_atoms.res_id.astype(str),
            bb_atoms.ins_code,
        ], axis=1),
        axis=0,
        return_inverse=True,
    )                                                          # (N_atoms,)

    n_res    = res_indices.max() + 1
    res_rmsf = np.zeros(n_res, dtype=float)
    counts   = np.zeros(n_res, dtype=int)

    np.add.at(res_rmsf, res_indices, atom_rmsf)
    np.add.at(counts,   res_indices, 1)
    res_rmsf /= counts                                        # mean per residue
    
    _, first_occurrence = np.unique(res_indices, return_index=True)
    original_order      = np.argsort(first_occurrence)
    res_rmsf            = res_rmsf[original_order]
    
    return res_rmsf[:, None]                                  # (L, 1)


#----------------------------
#from scipy.spatial import cKDTree


def get_surface_potential_sites(ab, scheme="imgt"):
    heavy = ab.heavy
    n_res = ab.L
    res_starts  = struc.get_residue_starts(heavy)
    atom_to_res = np.searchsorted(res_starts, np.arange(len(heavy)), side='right') - 1

    backbone_names = BACKBONE_ATOMS
    sidechain_mask = ~np.isin(heavy.atom_name, list(backbone_names))

    # Per-atom SASA — already computed in Ab.__init__
    atom_sasa = np.asarray(ab.sasa_atom, dtype=float)

    # ── polar atoms (non-carbon side-chain in polar/charged residues) ─────
    polar_residues   = POLAR
    charged_residues = CHARGED
    res_name_mask    = np.isin(heavy.res_name, list(polar_residues | charged_residues))
    polar_mask       = res_name_mask & (heavy.element != "C") & sidechain_mask

    # ── H-bond donors / acceptors ─────────────────────────────────────────
    donor_mask = np.zeros(len(heavy), dtype=bool)
    for resname, atom_names in HBOND_DONORS.items():
        for aname in atom_names:
            donor_mask |= (heavy.res_name == resname) & (heavy.atom_name == aname)

    acceptor_mask = np.zeros(len(heavy), dtype=bool)
    for resname, atom_names in HBOND_ACCEPTORS.items():
        for aname in atom_names:
            acceptor_mask |= (heavy.res_name == resname) & (heavy.atom_name == aname)

    # ── aggregate: sum atom SASA, masked by atom type ─────────────────────
    # Each atom contributes its own SASA (0 if buried) — no double counting.
    res_polar     = np.zeros(n_res, dtype=float)
    res_donors    = np.zeros(n_res, dtype=float)
    res_acceptors = np.zeros(n_res, dtype=float)

    np.add.at(res_polar,     atom_to_res, polar_mask    * atom_sasa)
    np.add.at(res_donors,    atom_to_res, donor_mask    * atom_sasa)
    np.add.at(res_acceptors, atom_to_res, acceptor_mask * atom_sasa)
    
    # ── CDR mask ──────────────────────────────────────────────────────────
    """
    cdr_mask = np.asarray(get_cdrs_fla_precise(heavy, scheme=scheme)).ravel()
    res_polar     *= cdr_mask
    res_donors    *= cdr_mask
    res_acceptors *= cdr_mask
    """
    # ── stack: [polar_sasa, donor_sasa, acceptor_sasa] ────────────────────
    return np.stack([res_polar, res_donors, res_acceptors], axis=1)
#-----------------


def cys_bridges(ab, cutoff=2.5):
    L = ab.L
    arr = np.zeros((L, 2), dtype=np.float32)

    # pass 1: mark every CYS and collect SG coordinates
    cys_info = []                      # [(residue_index, sg_xyz), ...]
    for i, res in enumerate(ab.residues):
        if res.res_name[0] != "CYS":
            continue
        arr[i, 0] = 1.0                           # channel 0: any Cys
        sg_mask = res.atom_name == "SG"
        if sg_mask.any():
            cys_info.append((i, res.coord[sg_mask][0]))

    # pass 2: mark Cys pairs whose SG atoms are within cutoff
    for a in range(len(cys_info)):
        i_a, sg_a = cys_info[a]
        for b in range(a + 1, len(cys_info)):
            i_b, sg_b = cys_info[b]
            if np.linalg.norm(sg_a - sg_b) < cutoff:
                arr[i_a, 1] = 1.0                 # channel 1: bonded Cys
                arr[i_b, 1] = 1.0

    return arr

#------------------

CDR_KEYS = ("cdr1_H", "cdr2_H", "cdr3_H", "cdr1_L", "cdr2_L", "cdr3_L")

def cdr_fw_contact_counts(ab, cdr_arrays, polar_cutoff=3.5, count="polar"):
    """
    Per-residue CDR-framework contact counts as a (ab.L, 1) array.
    Vectorized version. Drop-in replacement for the original.
    """
    keys = ("cdr1_H", "cdr2_H", "cdr3_H", "cdr1_L", "cdr2_L", "cdr3_L")
    L = ab.L

    # ---- 1. Combine 6 per-residue CDR masks ----
    cdr_per_res = np.zeros(L, dtype=bool)
    for k in keys:
        a = np.asarray(cdr_arrays[k], dtype=bool).reshape(-1)
        if a.shape[0] != L:
            raise ValueError(f"{k} has length {a.shape[0]}, expected ab.L = {L}")
        cdr_per_res |= a

    # ---- 2. Protonate if needed (do this once, all subsequent indexing is on prot_atoms) ----
    atoms = ab.atom_array
    if not bool(np.any(atoms.element == "H")):
        prot_atoms, _ = protonation(atoms)
    else:
        prot_atoms = atoms

    # ---- 3. Build per-atom residue index over prot_atoms (one pass via residue starts) ----
    starts = struc.get_residue_starts(prot_atoms, add_exclusive_stop=True)
    n_res  = len(starts) - 1
    atom_res_idx = np.repeat(np.arange(n_res, dtype=np.int32), np.diff(starts))

    # Residue identity (chain_id, res_id [, ins_code]) -> residue index in prot_atoms
    has_ins = "ins_code" in prot_atoms.get_annotation_categories()
    chain_pr = prot_atoms.chain_id[starts[:-1]]
    rid_pr   = prot_atoms.res_id  [starts[:-1]]
    ins_pr   = prot_atoms.ins_code[starts[:-1]] if has_ins else None

    if has_ins:
        res_key_to_idx = {
            (str(chain_pr[i]), int(rid_pr[i]), str(ins_pr[i])): i
            for i in range(n_res)
        }
    else:
        res_key_to_idx = {
            (str(chain_pr[i]), int(rid_pr[i])): i
            for i in range(n_res)
        }

    # ---- 4. Map ab.residues index -> prot_atoms residue index (single pass) ----
    ab_to_res = np.full(L, -1, dtype=np.int32)
    for i, res in enumerate(ab.residues):
        if has_ins:
            key = (str(res.chain_id[0]), int(res.res_id[0]), str(res.ins_code[0]))
        else:
            key = (str(res.chain_id[0]), int(res.res_id[0]))
        idx = res_key_to_idx.get(key, -1)
        ab_to_res[i] = idx

    # ---- 5. Per-atom CDR mask via np.isin on residue indices (no per-residue loop) ----
    cdr_res_in_pr = ab_to_res[cdr_per_res & (ab_to_res >= 0)]
    cdr_atom_mask = np.isin(atom_res_idx, cdr_res_in_pr)

    is_aa  = ~prot_atoms.hetero
    cdr_aa = cdr_atom_mask & is_aa
    fw_aa  = (~cdr_atom_mask) & is_aa

    n_polar  = np.zeros(n_res, dtype=np.int32)
    n_hbond  = np.zeros(n_res, dtype=np.int32)

    # ---- 6. Polar contacts: bulk pairwise distance, no per-atom loop ----
    if count in ("polar", "both"):
        is_polar = np.isin(prot_atoms.element, ("N", "O", "S"))
        cdr_polar_idx = np.where(cdr_aa & is_polar)[0]
        fw_polar_idx  = np.where(fw_aa  & is_polar)[0]

        if cdr_polar_idx.size and fw_polar_idx.size:
            # cdist: (Ncdr, Nfw) — for a Fab this is ~100 x ~500, microseconds.
            within = cdist(
                prot_atoms.coord[cdr_polar_idx],
                prot_atoms.coord[fw_polar_idx],
            ) < polar_cutoff
            per_cdr_atom = within.sum(axis=1, dtype=np.int32)        # contacts per CDR polar atom
            np.add.at(n_polar, atom_res_idx[cdr_polar_idx], per_cdr_atom)

    # ---- 7. H-bonds: biotite call + vectorized attribution ----
    if count in ("hbond", "both"):
        triplets = struc.hbond(prot_atoms, selection1=cdr_aa, selection2=fw_aa)
        if triplets.size:
            don_idx = triplets[:, 0]
            acc_idx = triplets[:, 2]
            cdr_idx = np.where(cdr_aa[don_idx], don_idx, acc_idx)    # whichever side is CDR
            np.add.at(n_hbond, atom_res_idx[cdr_idx], 1)

    if   count == "polar": n_per_res = n_polar
    elif count == "hbond": n_per_res = n_hbond
    elif count == "both":  n_per_res = n_polar + n_hbond
    else:                  raise ValueError(f"unknown count={count!r}")

    # ---- 8. Align back to ab.residues via fancy indexing (no loop) ----
    counts = np.zeros((L, 1), dtype=np.int32)
    valid = ab_to_res >= 0
    counts[valid, 0] = n_per_res[ab_to_res[valid]]
    return counts

#-----------------------------------
def build_ablang_index_map(Ab, vh_seq_passed_to_ablang, vl_seq_passed_to_ablang, has_separator=False):
    """
    Returns:
      biotite_to_ablang: np.ndarray[Ab.L], int, -1 if residue not in AbLang input
      ablang_to_biotite: np.ndarray[L_ab], int, -1 if AbLang position is separator/pad
    """
    if not hasattr(Ab, "_imgt_numbers"):
        chains, _, full_seq = _residue_metadata(Ab)
        Ab._imgt_numbers = _compute_imgt_numbers(Ab, chains, full_seq)
    
    chains, _, full_seq = _residue_metadata(Ab)
    
    L_vh = len(vh_seq_passed_to_ablang)
    L_vl = len(vl_seq_passed_to_ablang)
    sep = 1 if has_separator else 0
    L_ab = L_vh + sep + L_vl
    
    biotite_to_ablang = -np.ones(Ab.L, dtype=int)
    ablang_to_biotite = -np.ones(L_ab, dtype=int)
    
    # Walk biotite residues; whenever we see an Fv residue, assign next AbLang slot
    vh_counter = 0
    vl_counter = 0
    for i, (chain_letter, imgt_num) in enumerate(Ab._imgt_numbers):
        if chain_letter == "H" and vh_counter < L_vh:
            ab_pos = vh_counter
            biotite_to_ablang[i] = ab_pos
            ablang_to_biotite[ab_pos] = i
            vh_counter += 1
        elif chain_letter in ("L", "K") and vl_counter < L_vl:
            ab_pos = L_vh + sep + vl_counter
            biotite_to_ablang[i] = ab_pos
            ablang_to_biotite[ab_pos] = i
            vl_counter += 1
    
    # Sanity check
    if vh_counter != L_vh:
        print(f"[warn] VH coverage mismatch: biotite found {vh_counter}, AbLang expected {L_vh}")

    if vl_counter != L_vl:
        print(f"[warn] VL coverage mismatch: biotite found {vl_counter}, AbLang expected {L_vl}")

    
    return biotite_to_ablang[:, None], ablang_to_biotite[:, None]


#-------------------------------------
from scipy.spatial import cKDTree

def compute_surface_patch_features(Ab, radius=8.0, min_neighbors=5):
    """
    Per-residue patch features that distinguish flat/protruding/buried.
    
    Key metric: signed convexity = how far the CA is above (positive) or
    below (negative) the local plane, oriented away from antibody centroid.
      ~0     → flat surface
      large+ → protruding (bump out)
      large- → buried (cavity, or CA below surface)
    """
    coords = Ab.heavy.coord
    tree = cKDTree(coords)
    com = coords.mean(0)
    
    flatness   = np.zeros(Ab.L)
    convexity  = np.zeros(Ab.L)         # signed: + protrusion, 0 flat, - depression
    patch_area = np.zeros(Ab.L)
    n_neigh    = np.zeros(Ab.L, dtype=int)
    is_valid   = np.zeros(Ab.L, dtype=np.float32)
    
    for ri in range(Ab.L):
        ca = Ab.ca_coords[ri]
        if np.isnan(ca).any():
            continue
        idx = tree.query_ball_point(ca, r=radius)
        if len(idx) < min_neighbors:
            continue
        
        patch = coords[idx]
        center = patch.mean(0)
        centered = patch - center
        cov = centered.T @ centered / len(patch)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        if eigvals[-1] < 1e-6:
            continue
        
        # Plane normal = smallest-eigenvalue eigenvector
        normal = eigvecs[:, 0]
        # Orient outward (away from antibody center of mass)
        normal *= np.sign(np.dot(normal, ca - com))
        
        # Signed perpendicular distance of CA to the patch plane
        # >0  CA above plane (protrudes outward)
        # ~0  CA on plane (flat surface)
        # <0  CA below plane (buried / in cavity)
        convexity[ri]  = float(np.dot(ca - center, normal))
        flatness[ri]   = eigvals[0] / eigvals[-1]
        n_neigh[ri]    = len(idx)
        patch_area[ri] = float(np.pi * np.sqrt(eigvals[1] * eigvals[2]))
        is_valid[ri]   = 1.0
    
    return {
        "flatness":      flatness,
        "convexity":     convexity,        # the real curvature signal
        "patch_area":    patch_area,
        "n_neighbors":   n_neigh,
        "is_valid_patch": is_valid,
    }
#------------------
def hydrophobic_flat_score(Ab, radius=8.0, sigma=2.0):
    """
    High score = residue is on a flat hydrophobic surface.
    Low score = residue is protruding, buried, or hydrophilic.
    """
    p = compute_surface_patch_features(Ab, radius=radius)
    
    # Flatness: Gaussian kernel around convexity = 0
    # sigma controls how strict "flat" is (smaller = stricter)
    flatness_kernel = np.exp(-p["convexity"] ** 2 / (2 * sigma ** 2))
    
    # Combine: flat × hydrophobic × exposed
    # Per-residue patch hydrophobicity
    tree = cKDTree(Ab.ca_coords)
    patch_hydro = np.zeros(Ab.L)
    for ri in range(Ab.L):
        nb = tree.query_ball_point(Ab.ca_coords[ri], r=radius)
        patch_hydro[ri] = float(Ab.hydros[ri]*(Ab.hydros[nb]* flatness_kernel[nb]*Ab.sasa_resi[nb]).sum())
    
    #score = flatness_kernel * flatness_kernel *  Ab.sasa_resi  #exposures
    score = flatness_kernel  *  Ab.sasa_resi  #exposures
    score = score * p["is_valid_patch"]
    score_raw = flatness_kernel  *  flatness_kernel  * Ab.sasa_resi  #exposures
    
    return {
        "convexity":p["convexity"],
        "flatness_kernel":   flatness_kernel,
        "patch_hydro":       patch_hydro,
        "patch_flat":  score,
        "patch_flat_raw":  score_raw
    }

#---------------------------



def compute_ionic_contacts_split(
    Ab,
    distance_threshold=4.0,
    min_seq_separation=2,
    include_his=False,
    sentinel=0,
):
    """
    For each residue, split charged-atom scores into two categories:
      bonded_score:    MAX over bonded charged atoms of (100 if fully buried,
                       else  SASA). Range 0–100; higher = the most-buried
                       bonded atom in this residue is more buried.
      unused_score:    sum SASA of charged atoms NOT in any salt bridge

    Sentinel value (default 0):
      - bonded_score[i] = sentinel if residue i has no charged side-chain atoms
      - unused_score[i] = sentinel if residue i has no charged side-chain atoms
    For residues WITH charged atoms:
      - bonded_score = 0 means atoms are present but none are in a salt bridge
                          OR all bonded atoms are fully buried (SASA = 0)
      - unused_score = 0 means atoms are present but all are in salt bridges
                          OR all unused atoms are fully buried (SASA = 0)
    
    Args:
        Ab:                  ab_structure with .heavy and .L
        distance_threshold:  max heavy-atom distance for a salt bridge (default 4.0 Å)
        min_seq_separation:  skip same/adjacent residues
        include_his:         treat His ND1/NE2 as positive
        atom_sasa:           pre-computed per-atom SASA. If None, computed via biotite.
        sentinel:            value used for residues without charged atoms
    
    Returns:
        dict with arrays of shape [Ab.L]:
          bonded_score:  SASA sum of bonded charged atoms (or sentinel)
          unused_score:  SASA sum of unbonded charged atoms (or sentinel)
          n_bonded:      number of distinct charged atoms in salt bridges
          n_unused:      number of distinct charged atoms NOT in salt bridges
          has_charged:   bool, True if residue has charged side-chain atoms
    """
    heavy = Ab.heavy
    coords = heavy.coord
    atom_sasa = Ab.sasa_atom

    # Build pos/neg atom criteria
    pos_atoms = dict(POSITIVE_ATOMS)
    if include_his:
        pos_atoms["HIS"] = ["ND1", "NE2"]

    pos_mask = np.zeros(len(heavy), dtype=bool)
    neg_mask = np.zeros(len(heavy), dtype=bool)

    for i in range(len(heavy)):
        rname = heavy.res_name[i]
        aname = heavy.atom_name[i]
        if rname in pos_atoms and aname in pos_atoms[rname]:
            pos_mask[i] = True
        elif rname in NEGATIVE_ATOMS and aname in NEGATIVE_ATOMS[rname]:
            neg_mask[i] = True

    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(neg_mask)[0]
    charged_mask = pos_mask | neg_mask

    # Per-atom SASA via biotite if not provided
    if atom_sasa is None:
        atom_sasa = struc.sasa(heavy, probe_radius=1.4, vdw_radii="ProtOr")
    atom_sasa = np.nan_to_num(np.asarray(atom_sasa, dtype=float), nan=0.0)

    # Map atoms to residues
    res_starts = struc.get_residue_starts(heavy)
    atom_to_res = np.zeros(len(heavy), dtype=int)
    for i, start in enumerate(res_starts):
        end = res_starts[i+1] if i+1 < len(res_starts) else len(heavy)
        atom_to_res[start:end] = i

    # Which residues have any charged side-chain atoms?
    has_charged = np.zeros(Ab.L, dtype=bool)
    for ai in np.where(charged_mask)[0]:
        has_charged[atom_to_res[ai]] = True

    # Find salt-bridge atom pairs
    bonded_atoms = set()

    if len(pos_idx) > 0 and len(neg_idx) > 0:
        pos_tree = cKDTree(coords[pos_idx])
        neg_tree = cKDTree(coords[neg_idx])
        pairs = pos_tree.query_ball_tree(neg_tree, r=distance_threshold)

        for pi, neg_local in enumerate(pairs):
            if not neg_local:
                continue
            atom_p = pos_idx[pi]
            res_p = atom_to_res[atom_p]
            for nj in neg_local:
                atom_n = neg_idx[nj]
                res_n = atom_to_res[atom_n]
                if abs(res_p - res_n) < min_seq_separation:
                    continue
                bonded_atoms.add(int(atom_p))
                bonded_atoms.add(int(atom_n))

    # Aggregate per residue
    bonded_score = np.zeros(Ab.L)
    unused_score = np.zeros(Ab.L)
    n_bonded     = np.zeros(Ab.L, dtype=int)
    n_unused     = np.zeros(Ab.L, dtype=int)

    charged_atom_indices = np.where(charged_mask)[0]
    for ai in charged_atom_indices:
        ri = atom_to_res[ai]
        if int(ai) in bonded_atoms:
            # 100 if fully buried, else 100 - sasa; take max across the residue
            atom_score = max(0.0, 100.0 - atom_sasa[ai])
            if atom_score > bonded_score[ri]:
                bonded_score[ri] = atom_score
            n_bonded[ri] += 1
        else:
            unused_score[ri] += atom_sasa[ai]
            n_unused[ri] += 1

    # Apply sentinel for residues without any charged atoms
    bonded_score[~has_charged] = sentinel
    unused_score[~has_charged] = sentinel

    return {
        "bonded_score":  bonded_score,
        "unused_score":  unused_score,
        "n_bonded":      n_bonded,
        "n_unused":      n_unused,
        "has_charged":   has_charged,
    }

#---------

def per_residue_chain_charge(Ab):
    """
    Per-residue array of length Ab.L:
      - residues in chain H get the total charge of chain H
      - residues in chain L get the total charge of chain L
    Useful as a broadcast feature for per-residue models.
    """
    starts    = struc.get_residue_starts(Ab.heavy)
    chain_ids = Ab.heavy.chain_id[starts]

    charge_H = float(Ab.charges[chain_ids == "H"].sum())
    charge_L = float(Ab.charges[chain_ids == "L"].sum())

    arr = np.zeros(Ab.L, dtype=float)
    arr[chain_ids == "H"] = charge_H
    arr[chain_ids == "L"] = charge_L
    return arr
#-------------------------
def get_chain_charges(Ab):
    """Formal net charge of VH and VL as two per-molecule globals."""
    starts    = struc.get_residue_starts(Ab.heavy)
    chain_ids = Ab.heavy.chain_id[starts]
    charge_H = float(Ab.charges[chain_ids == "H"].sum())
    charge_L = float(Ab.charges[chain_ids != "H"].sum())   # light = anything not H
    return charge_H, charge_L
#=---------------------

def _residue_charge_vec(res_names):
    q = np.zeros(len(res_names), dtype=int)
    q[np.isin(res_names, list(POSITIVE))] = +1
    q[np.isin(res_names, list(NEGATIVE))] = -1
    return q


def compute_charged_patch_features(
    ab,
    radii=[6.0, 8.0, 10.0],
    exposure_frac: float = 0.10,
    use_sidechain_sasa: bool = True,
):
    """
    Per-residue charged-patch features at multiple neighborhood radii.

    A residue counts as "exposed" if its (side-chain) SASA >= exposure_frac * ab.ref_sasa.
    Only exposed charged residues contribute to neighborhood sums.

    Returns
    -------
    dict with:
        pos_area     : (L, n_radii) Σ exposed SC SASA of +charged neighbors
        neg_area     : (L, n_radii) Σ exposed SC SASA of -charged neighbors
        net_area     : (L, n_radii) pos_area - neg_area
        total_area   : (L, n_radii) pos_area + neg_area
        n_pos        : (L, n_radii) count of + charged exposed neighbors
        n_neg        : (L, n_radii) count of - charged exposed neighbors
    """
    radii = tuple(float(r) for r in radii)

    # ---- per-residue arrays consistent with ab.L --------------------------
    starts = struc.get_residue_starts(ab.heavy)
    assert len(starts) == ab.L, f"residue count mismatch: starts={len(starts)} ab.L={ab.L}"
    ends   = np.append(starts[1:], len(ab.heavy))

    res_name = ab.heavy.res_name[starts]
    sasa     = ab.sasa_sch if use_sidechain_sasa else ab.sasa_resi
    ref_table = ab.ref_sasa  # dict like {"ALA": 67.0, "ARG": 241.0, ...}
    ref_sasa  = np.array([float(ref_table.get(r, np.nan)) for r in res_name], dtype=float)
    assert sasa.shape[0] == ab.L and ref_sasa.shape[0] == ab.L

    # CA coordinates (fallback to first heavy atom if CA missing)
    is_ca  = ab.heavy.atom_name == "CA"
    coord  = ab.heavy.coord
    ca_xyz = np.empty((ab.L, 3), dtype=float)
    for ri, (s, e) in enumerate(zip(starts, ends)):
        m = is_ca[s:e]
        ca_xyz[ri] = coord[s:e][m][0] if m.any() else coord[s]

    # ---- charge + exposure masks -----------------------------------------
    own_charge = _residue_charge_vec(res_name)
    is_charged = own_charge != 0

    # exposure threshold = exposure_frac * ref_sasa (guard against zero ref)
    ref_safe = np.where(np.isfinite(ref_sasa) & (ref_sasa > 0), ref_sasa, np.inf)
    thresh   = exposure_frac * ref_safe
    is_exposed         = sasa >= thresh
    is_exposed_charged = is_charged & is_exposed

    contrib_pos = np.where((own_charge == +1) & is_exposed, sasa, 0.0)
    contrib_neg = np.where((own_charge == -1) & is_exposed, sasa, 0.0)

    # ---- multi-radius neighborhood sums via one cKDTree ------------------
    tree     = cKDTree(ca_xyz)
    r_max    = max(radii)
    nbrs_all = tree.query_ball_point(ca_xyz, r=r_max)
    # also need pairwise distances to bin into smaller radii
    n_r = len(radii)
    pos_area = np.zeros((ab.L, n_r))
    neg_area = np.zeros((ab.L, n_r))
    n_pos    = np.zeros((ab.L, n_r), dtype=int)
    n_neg    = np.zeros((ab.L, n_r), dtype=int)

    for i, nbrs in enumerate(nbrs_all):
        if len(nbrs) <= 1:
            continue
        nbrs = np.fromiter((j for j in nbrs if j != i), dtype=int)
        if nbrs.size == 0:
            continue
        d  = np.linalg.norm(ca_xyz[nbrs] - ca_xyz[i], axis=1)
        cp = contrib_pos[nbrs]
        cn = contrib_neg[nbrs]
        for k, r in enumerate(radii):
            in_r = d <= r
            if not in_r.any():
                continue
            pos_area[i, k] = cp[in_r].sum()
            neg_area[i, k] = cn[in_r].sum()
            n_pos[i, k]    = int((cp[in_r] > 0).sum())
            n_neg[i, k]    = int((cn[in_r] > 0).sum())

    return {
        "pos_area":           pos_area,
        "neg_area":           neg_area,
        "net_area":           pos_area - neg_area,
        "total_area":         pos_area + neg_area,
        "n_pos":              n_pos,
        "n_neg":              n_neg,
    }

#--------------------
#--------------  pI
import propka.run as pk
import logging
logging.getLogger("propka").setLevel(logging.CRITICAL)  

def fv_pi(pdb_path, ph_list=(7.5, 6.0)):
    """
    Structure-based pI and net charge at each pH in ph_list.
    Requires: pip install propka
    """
    mol = pk.single(str(pdb_path), optargs=["--quiet"], write_pka=False)
    pi_folded, pi_unfolded = mol.get_pi()

    conf = mol.conformations["AVR"]

    out = {
        "pI_folded":   round(pi_folded,   2),
        "pI_unfolded": round(pi_unfolded, 2),
    }

    for ph in ph_list:
        try:
            q = conf.calculate_charge(mol.version.parameters, ph=ph)
        except TypeError:
            q = conf.calculate_charge(mol.version.parameters, ph)
        net_charge = sum(q) if isinstance(q, (list, tuple)) else float(q)
        out[f"net_charge_ph{ph}"] = round(net_charge, 2)

    return out

#----
import warnings

def get_light_isotype(Ab):
    """
    Per-antibody light-chain isotype as a global scalar feature:
        1.0 = lambda,  0.0 = kappa,  nan = no light chain detected.
    Reuses the ANARCI chain type captured during IMGT numbering (no extra ANARCI run).
    """
    if not hasattr(Ab, "_imgt_numbers"):
        chains, res_ids, full_seq = _residue_metadata(Ab)
        Ab._imgt_numbers = _compute_imgt_numbers(Ab, chains, full_seq)
    iso = getattr(Ab, "_light_isotype", None)
    if iso == "L":
        return 1.0
    if iso == "K":
        return 0.0

    ab_id = getattr(Ab, "Ab_ID", getattr(Ab, "name", "?"))
    warnings.warn(f"get_light_isotype: no light chain detected for Ab={ab_id}",
                  RuntimeWarning, stacklevel=2)
    return float("nan")