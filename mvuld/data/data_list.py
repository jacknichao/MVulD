# from __future__ import print_function, division

from importlib.resources import path
import torch
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from sklearn.preprocessing import StandardScaler
import random
from dgl.data.utils import load_graphs, save_graphs
import torch.utils.data as data
import os
import os.path
import pandas as pd
import torch as th
import dgl
from tqdm import tqdm
from glob import glob
from collections import Counter
from models.unixcoder import UniXcoder,MyUniXcoder

import sys
path1 = os.path.dirname(sys.path[0])
sys.path.append(path1)
import sastvd as svd
import sastvd.helpers.joern as svdj
import pickle

type_map = {'UNKNOWN': 0, 'METHOD': 1, 'METHOD_PARAMETER_IN': 2, 'BLOCK': 3, 'External Function Call': 4,
            'Comparison Operator': 5, 'IDENTIFIER': 6, 'Assignment Operator': 7, 'RETURN': 8, 'LITERAL': 9,
            'METHOD_RETURN': 10, 'METHOD_PARAMETER_OUT': 11, 'IF': 12, 'Arithmetic Operator': 13,
            'Builtin Function Call': 14, 'Access Operator': 15, 'FIELD_IDENTIFIER': 16, 'Other Operator': 17,
            'LOCAL': 18, 'Logical Operator': 19, 'Cast Operator': 20, 'WHILE': 21, 'ELSE': 22, 'FOR': 23, 'GOTO': 24,
            'JUMP_TARGET': 25, 'SWITCH': 26, 'BREAK': 27, 'DO': 28, 'CONTINUE': 29, 'TYPE_DECL': 30, 'MEMBER': 31}

type_one_hot = np.eye(len(type_map))

def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def pil_loader(path):
    with open(path, 'rb') as f: 
        with Image.open(f) as img:
            return img.convert('RGB')
            # return Image.open(path).convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    # else:
    return pil_loader(path)


class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        ** imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, gtype="all",labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(image_list, labels)
        # print(len(imgs))
        self.imgs = imgs
        self.graph_type = gtype
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        func_emb_unix_path = self.get_unixfuncEmb_path()
        unix_func_emb_df = pd.read_pickle(func_emb_unix_path)
        self.df = unix_func_emb_df

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index] 
        img = self.loader(path)
        self.preimg = img 
        if self.transform is not None:
            img = self.transform(img) 
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        _id = int(path.split('/')[-1].rstrip('.png')) # str to int
        g = self.item(_id) 
        # print(g)
        id_list = th.Tensor([_id]).float()
        vuln=th.Tensor([target]).float()
        # g.ndata["_FLABLE"] = vuln.max().repeat((g.number_of_nodes())) 
        # g.ndata["_id"] = id_list.max().repeat((g.number_of_nodes())) 
        
        g.ndata["_IMG"]=img.repeat((g.number_of_nodes(), 1,1,1)) 
        # print(g.ndata["_IMG"].shape)
        
        swin_path = svd.cache_dir() / f"swinv2_method_level_try5/{_id}.pt" # swin feature
        img_embedding=th.load(swin_path)
        # g.ndata["_IMG_EMB"] = th.load(swin_path).repeat((g.number_of_nodes(), 1))
        ## img= g.ndata["_IMG"][0]
        
        ## text embedding
        func_text_embedding= self.df[self.df.ids==_id].iloc[0].repr
        func_text_embedding = torch.tensor(func_text_embedding)
        return g,img_embedding,func_text_embedding,target

    def __len__(self):
        return len(self.imgs)
    
    def get_pos_emb_with_id(self,_id):
        '''use _id to obtain pos dict obtained by ocr: (id,[startX,endX,endX,endY]]'''
        pos_dict_path = '/data1/username/project/MMVD/mmvd/datasets/norm_pos_dict'
        pos_dict_file = os.path.join(pos_dict_path, f"{_id}.pkl")
        print(pos_dict_file)
        with open(pos_dict_file, "rb") as tf:
            pd = pickle.load(tf)
        return pd
        
    def cache_g_items(self,model):
        """Cache all graph items."""
        for idx in tqdm(range(len(self.imgs)),desc="cache_g"): 
        # for idx in range(len(self.imgs)):
            path, _ = self.imgs[idx] 
            _id = int(path.split('/')[-1].rstrip('.png'))
            try:
                self.item(_id, model)
            except Exception as E:
                print(E)

    def cache_items_df(self,model):
        """Cache all items."""
        print("============= Cache all items =============")
        balanced_df_path = '/data1/username/project/MMVD/baselines/storage/cache/data/bigvul/bigvul_cleaned3_balanced.pkl'
        cache_path = balanced_df_path
        df = pd.read_pickle(cache_path)
        print(df.info())
        for i in tqdm(df.sample(len(df))._id.tolist()):
            try:
                self.item(i, model)
            except Exception as E:
                print(E)
        
    def cache_swin_features(self, swin): 
        """Cache imgs features using swinv model.
        ONLY NEEDS TO BE RUN ONCE.
        """
        savedir = svd.get_dir(svd.cache_dir() / "swinv2_method_level_try5") #
        done = [int(i.split("/")[-1].split(".")[0]) for i in glob(str(savedir / "*"))]
        done = set(done)
        print(savedir)
        batch_size=8 
        batches = svd.chunks((range(len(self.imgs))), batch_size) 
        print(batches)
        for idx_batch in tqdm(batches):
            temp= self.imgs[idx_batch[0] : idx_batch[-1] + 1]
            batch_path = [i[0] for i in temp] # 
            batch_target = [i[1] for i in temp] # 
            
            batch_ids = [int(path.split('/')[-1].rstrip('.png')) for path in batch_path]
            
            batch_imgs = [self.transform(self.loader(path)).tolist() for path in batch_path]
            batch_imgs= th.tensor(batch_imgs)
            # print(batch_imgs.shape) # batch_imgs type:list batch_imgs[0] type:tensor
            if set(batch_ids).issubset(done):
                # print("swin features already exist")
                continue
            img_feature=swin.forward_features(batch_imgs).detach().cpu()

            print(img_feature.shape) # batch_size x 1024
            assert len(batch_imgs) == len(batch_ids)
            for i in range(len(batch_imgs)):
                # print(batch_ids[i])
                # print(img_feature[i]) # shape:[1,1024]
                # print(savedir / f"{batch_ids[i]}.pt")
                th.save(img_feature[i], savedir / f"{batch_ids[i]}.pt")
                 
    def itempath(self,_id): 
        """Get itempath path from item id."""
        item_path = f"/data1/username/project/MMVD/baselines/storage/processed/bigvul/func_before/{_id}.c"
        # return svd.processed_dir() / f"bigvul/func_before/{_id}.c" 
        return item_path

    def get_unixfuncEmb_path(self):
        balanced_path = '/data1/username/project/MMVD/baselines/storage/cache/unixcoder_output/bigvul/result.pkl'
        path = balanced_path
        return path

    def cache_g_items_tokenids(self,tokenize): 
        """Cache all items."""
        for idx in tqdm(range(len(self.imgs)),desc="cache_tokenids"): # 
        # for idx in range(len(self.imgs)):
            path, _ = self.imgs[idx] 
            _id = int(path.split('/')[-1].rstrip('.png'))
            try:
                self.item1(_id, tokenize)
            except Exception as E:
                print(E)

    def item1(self,_id,tokenize=None):
        """Cache item."""
        savedir = svd.get_dir(
            # svd.cache_dir() / f"bigvul_gitem_unix_tokenids_{self.graph_type}"
            svd.cache_dir() / f"bigvul_gitem_unix_tokenids_64_{self.graph_type}"
        ) / str(_id)
        if os.path.exists(savedir):
            g = load_graphs(str(savedir))[0][0]
            # print(g)
            return g 
        code, lineno,nt,ei, eo, et = feature_extraction(
            self.itempath(_id), self.graph_type
        )
        g = dgl.graph((eo, ei)) 
        if tokenize: # unxicoder tokenize()
            sents = [c.replace("\\t", "").replace("\\n", "") for c in code]
            ## save token ids
            source_ids=[]
            for node in sents:
                node_str = ' '.join(node.split()) # example[0]：funcs [1]:lables [2]:ids
                node_ids = tokenize([node_str], max_length=64 , padding=True)
                source_ids.append(node_ids[0])
            all_source_ids = torch.tensor([i for i in source_ids]).detach().cpu()
            # print(all_source_ids)
            g.ndata["_token_ids"] = all_source_ids #[node num,,64]
        g.edata["_ETYPE"] = th.tensor(et).long()
        g = dgl.add_self_loop(g)
        save_graphs(str(savedir), [g])
        return g   

    def item(self, _id,model=None): 
        # print(_id)
        """Cache item."""
        savedir = '/data1/username/project/MMVD/mmvd/datasets/bigvul_gitem_unix' # unix(batch_size=4,best-f1)
        savedir = svd.get_dir(os.path.join(savedir, f"{self.graph_type}"))
        savedir = os.path.join(savedir,str(_id))

        if os.path.exists(savedir):
            g = load_graphs(str(savedir))[0][0]
            return g 
        
        code, lineno,nt,ei, eo, et = feature_extraction(
            self.itempath(_id), self.graph_type
        )
        g = dgl.graph((eo, ei)) 
        g.ndata["_lineno"] = torch.Tensor(lineno) 
        ## pos embedding 
        norm_pos_dict = self.get_pos_emb_with_id(_id) # obtain pos dict by _id
        node_bboxes = np.zeros((g.number_of_nodes(), 4)) # build bbox(d=4
        nid_list = lineno #  =g.ndata["_lineno"].tolist()
        for index,nid in enumerate(nid_list): 
            if int(nid) in norm_pos_dict : 
                node_bboxes[index] = norm_pos_dict[int(nid)]
            else:
                node_bboxes[index] = np.zeros((1, 4)) 
        g.ndata["pos_emb"] = torch.Tensor(node_bboxes)

        ## node embedding ：unix text embeddding
        if model: 
            print(f"==gtype={self.graph_type},id ={_id} unix embed==") 
            code = [c.replace("\\t", "").replace("\\n", "") for c in code]
            chunked_batches = svd.chunks(code, 128) 
            unix_features = [model.myEncode(c) for c in chunked_batches] 
            ## text_features
            g.ndata["_UNIX_NODE_EMB"] = th.cat(unix_features) 
            # text_feats = [uf.tolist() for uf in unix_features] #  [1,node num,768]
            # text_feats = np.squeeze(text_feats) # [node num,768] 
            ## ntype info:
            # structure_feats = [type_one_hot[type_map[node_type] - 1] for node_type in nt] # [node num,32]
            ## comprehensive node embedding：ntype+text
            # node_feats = np.concatenate([structure_feats, text_feats], axis=1) # [node num,768+32=800]
            # g.ndata["_ALL_NODE_EMB"] = torch.Tensor(np.array(node_feats))

        g.edata["_ETYPE"] = th.Tensor(et).long()
        # emb_path = svd.cache_dir() / f"codebert_method_level/{_id}.pt" 
        # g.ndata["_FUNC_EMB"] = th.load(emb_path).repeat((g.number_of_nodes(), 1))
        ## 保存func_emb_unix
        func_emb= self.df[self.df.ids==_id].iloc[0].repr
        g.ndata["_FUNC_EMB"] = torch.Tensor([func_emb] * g.number_of_nodes())
        g = dgl.add_self_loop(g)
        save_graphs(str(savedir), [g])
        # print(g.ndata)
        return g

def ne_groupnodes(n, e):
    """Group nodes with same line number."""
    nl = n[n.lineNumber != ""].copy()
    nl.lineNumber = nl.lineNumber.astype(int) 
    nl = nl.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
    nl = nl.groupby("lineNumber").head(1) 
    
    el = e.copy()
    el.innode = el.line_in
    el.outnode = el.line_out

    nl.id = nl.lineNumber
    nl = svdj.drop_lone_nodes(nl, el)

    el = el.drop_duplicates(subset=["innode", "outnode", "etype"]) 
    
    el = el[el.innode.apply(lambda x: isinstance(x, float))]
    el = el[el.outnode.apply(lambda x: isinstance(x, float))]
    el.innode = el.innode.astype(int)
    el.outnode = el.outnode.astype(int)
    return nl, el


def feature_extraction(_id, graph_type="all", return_nodes=False):
    """Extract graph feature (basic).
    _id = svddc.BigVulDataset.itempath(177775)
    return_nodes arg is used to get the node information (for empirical evaluation).
    """
    # Get graph nodes and edges
    n, e = svdj.get_node_edges(_id)
    n, e = ne_groupnodes(n, e)

    # Return node metadata
    if return_nodes:
        return n

    # Filter nodes
    e = svdj.rdg(e, graph_type.split("+")[0]) 
    n = svdj.drop_lone_nodes(n, e) 
    counter, label_info = count_labels(n) 
    ntypes = label_info
    # Plot graph
    # svdj.plot_graph_node_edge_df(n, e)

    # Map line numbers to indexing
    n = n.reset_index(drop=True).reset_index() 
    iddict = pd.Series(n.index.values, index=n.id).to_dict()
    e.innode = e.innode.map(iddict)
    e.outnode = e.outnode.map(iddict)

    # Map edge types
    
    etypes = [etype_2_id(t) for t in e.etype.tolist()]
    # etypes = e.etype.tolist()
    # d = dict([(y, x) for x, y in enumerate(sorted(set(etypes)))])
    # etypes = [d[i] for i in etypes]

    return n.code.tolist(), n.id.tolist(),ntypes,e.innode.tolist(), e.outnode.tolist(), etypes



def etype_2_id(etype):
    return etype_map[etype]

def type_2_type(info):
    for i in range(1):
        if info['_label'] == "CALL":
            if "<operator>" in info["name"]:
                if "assignment" in info["name"]:
                    new_type = "Assignment Operator"
                    continue
                if (
                        "addition" in info["name"]
                        or "subtraction" in info["name"]
                        or "division" in info["name"]
                        or "Plus" in info["name"]
                        or "Minus" in info["name"]
                        or "minus" in info["name"]
                        or "plus" in info["name"]
                        or "modulo" in info["name"]
                        or "multiplication" in info["name"]
                ):
                    new_type = "Arithmetic Operator"
                    continue
                if (
                        "lessThan" in info["name"]
                        or "greaterThan" in info["name"]
                        or "EqualsThan" in info["name"]
                        or "equals" in info["name"]
                ):
                    new_type = "Comparison Operator"
                    continue
                if (
                        "FieldAccess" in info["name"]
                        or "IndexAccess" in info["name"]
                        or "fieldAccess" in info["name"]
                        or "indexAccess" in info["name"]
                ):
                    new_type = "Access Operator"
                    continue
                if (
                        "logical" in info["name"]
                        or "<operator>.not" in info["name"]
                        or "<operator>.or" in info["name"]
                        or "<operator>.and" in info["name"]
                        or "conditional" in info["name"]
                ):
                    new_type = "Logical Operator"
                    continue
                if "<operator>.cast" in info["name"]:
                    new_type = "Cast Operator"
                    continue
                if "<operator>" in info["name"]:
                    new_type = "Other Operator"
                    continue
            elif info["name"] in l_funcs:
                new_type = "Builtin Function Call"
                continue
            else:
                new_type = "External Function Call"
                continue
        if info["_label"] == "CONTROL_STRUCTURE":
            new_type = info["controlStructureType"]
            continue
        new_type = info["_label"]
    return new_type
    
def count_labels(nodes):
    """Get info about nodes."""
    label_info = []
    for _, info in nodes.iterrows():
        new_type = type_2_type(info)
        label_info.append(new_type)

    counter = Counter(label_info)
    return counter, label_info

etype_map = {
    'AST': 0,
    'CDG': 1,
    'REACHING_DEF': 2,
    'CFG': 3,
    'EVAL_TYPE': 4,
    'REF': 5
}

l_funcs = set([
    "StrNCat",
    "getaddrinfo",
    "_ui64toa",
    "fclose",
    "pthread_mutex_lock",
    "gets_s",
    "sleep",
    "_ui64tot",
    "freopen_s",
    "_ui64tow",
    "send",
    "lstrcat",
    "HMAC_Update",
    "__fxstat",
    "StrCatBuff",
    "_mbscat",
    "_mbstok_s",
    "_cprintf_s",
    "ldap_search_init_page",
    "memmove_s",
    "ctime_s",
    "vswprintf",
    "vswprintf_s",
    "_snwprintf",
    "_gmtime_s",
    "_tccpy",
    "*RC6*",
    "_mbslwr_s",
    "random",
    "__wcstof_internal",
    "_wcslwr_s",
    "_ctime32_s",
    "wcsncat*",
    "MD5_Init",
    "_ultoa",
    "snprintf",
    "memset",
    "syslog",
    "_vsnprintf_s",
    "HeapAlloc",
    "pthread_mutex_destroy",
    "ChangeWindowMessageFilter",
    "_ultot",
    "crypt_r",
    "_strupr_s_l",
    "LoadLibraryExA",
    "_strerror_s",
    "LoadLibraryExW",
    "wvsprintf",
    "MoveFileEx",
    "_strdate_s",
    "SHA1",
    "sprintfW",
    "StrCatNW",
    "_scanf_s_l",
    "pthread_attr_init",
    "_wtmpnam_s",
    "snscanf",
    "_sprintf_s_l",
    "dlopen",
    "sprintfA",
    "timed_mutex",
    "OemToCharA",
    "ldap_delete_ext",
    "sethostid",
    "popen",
    "OemToCharW",
    "_gettws",
    "vfork",
    "_wcsnset_s_l",
    "sendmsg",
    "_mbsncat",
    "wvnsprintfA",
    "HeapFree",
    "_wcserror_s",
    "realloc",
    "_snprintf*",
    "wcstok",
    "_strncat*",
    "StrNCpy",
    "_wasctime_s",
    "push*",
    "_lfind_s",
    "CC_SHA512",
    "ldap_compare_ext_s",
    "wcscat_s",
    "strdup",
    "_chsize_s",
    "sprintf_s",
    "CC_MD4_Init",
    "wcsncpy",
    "_wfreopen_s",
    "_wcsupr_s",
    "_searchenv_s",
    "ldap_modify_ext_s",
    "_wsplitpath",
    "CC_SHA384_Final",
    "MD2",
    "RtlCopyMemory",
    "lstrcatW",
    "MD4",
    "MD5",
    "_wcstok_s_l",
    "_vsnwprintf_s",
    "ldap_modify_s",
    "strerror",
    "_lsearch_s",
    "_mbsnbcat_s",
    "_wsplitpath_s",
    "MD4_Update",
    "_mbccpy_s",
    "_strncpy_s_l",
    "_snprintf_s",
    "CC_SHA512_Init",
    "fwscanf_s",
    "_snwprintf_s",
    "CC_SHA1",
    "swprintf",
    "fprintf",
    "EVP_DigestInit_ex",
    "strlen",
    "SHA1_Init",
    "strncat",
    "_getws_s",
    "CC_MD4_Final",
    "wnsprintfW",
    "lcong48",
    "lrand48",
    "write",
    "HMAC_Init",
    "_wfopen_s",
    "wmemchr",
    "_tmakepath",
    "wnsprintfA",
    "lstrcpynW",
    "scanf_s",
    "_mbsncpy_s_l",
    "_localtime64_s",
    "fstream.open",
    "_wmakepath",
    "Connection.open",
    "_tccat",
    "valloc",
    "setgroups",
    "unlink",
    "fstream.put",
    "wsprintfA",
    "*SHA1*",
    "_wsearchenv_s",
    "ualstrcpyA",
    "CC_MD5_Update",
    "strerror_s",
    "HeapCreate",
    "ualstrcpyW",
    "__xstat",
    "_wmktemp_s",
    "StrCatChainW",
    "ldap_search_st",
    "_mbstowcs_s_l",
    "ldap_modify_ext",
    "_mbsset_s",
    "strncpy_s",
    "move",
    "execle",
    "StrCat",
    "xrealloc",
    "wcsncpy_s",
    "_tcsncpy*",
    "execlp",
    "RIPEMD160_Final",
    "ldap_search_s",
    "EnterCriticalSection",
    "_wctomb_s_l",
    "fwrite",
    "_gmtime64_s",
    "sscanf_s",
    "wcscat",
    "_strupr_s",
    "wcrtomb_s",
    "VirtualLock",
    "ldap_add_ext_s",
    "_mbscpy",
    "_localtime32_s",
    "lstrcpy",
    "_wcsncpy*",
    "CC_SHA1_Init",
    "_getts",
    "_wfopen",
    "__xstat64",
    "strcoll",
    "_fwscanf_s_l",
    "_mbslwr_s_l",
    "RegOpenKey",
    "makepath",
    "seed48",
    "CC_SHA256",
    "sendto",
    "execv",
    "CalculateDigest",
    "memchr",
    "_mbscpy_s",
    "_strtime_s",
    "ldap_search_ext_s",
    "_chmod",
    "flock",
    "__fxstat64",
    "_vsntprintf",
    "CC_SHA256_Init",
    "_itoa_s",
    "__wcserror_s",
    "_gcvt_s",
    "fstream.write",
    "sprintf",
    "recursive_mutex",
    "strrchr",
    "gethostbyaddr",
    "_wcsupr_s_l",
    "strcspn",
    "MD5_Final",
    "asprintf",
    "_wcstombs_s_l",
    "_tcstok",
    "free",
    "MD2_Final",
    "asctime_s",
    "_alloca",
    "_wputenv_s",
    "_wcsset_s",
    "_wcslwr_s_l",
    "SHA1_Update",
    "filebuf.sputc",
    "filebuf.sputn",
    "SQLConnect",
    "ldap_compare",
    "mbstowcs_s",
    "HMAC_Final",
    "pthread_condattr_init",
    "_ultow_s",
    "rand",
    "ofstream.put",
    "CC_SHA224_Final",
    "lstrcpynA",
    "bcopy",
    "system",
    "CreateFile*",
    "wcscpy_s",
    "_mbsnbcpy*",
    "open",
    "_vsnwprintf",
    "strncpy",
    "getopt_long",
    "CC_SHA512_Final",
    "_vsprintf_s_l",
    "scanf",
    "mkdir",
    "_localtime_s",
    "_snprintf",
    "_mbccpy_s_l",
    "memcmp",
    "final",
    "_ultoa_s",
    "lstrcpyW",
    "LoadModule",
    "_swprintf_s_l",
    "MD5_Update",
    "_mbsnset_s_l",
    "_wstrtime_s",
    "_strnset_s",
    "lstrcpyA",
    "_mbsnbcpy_s",
    "mlock",
    "IsBadHugeWritePtr",
    "copy",
    "_mbsnbcpy_s_l",
    "wnsprintf",
    "wcscpy",
    "ShellExecute",
    "CC_MD4",
    "_ultow",
    "_vsnwprintf_s_l",
    "lstrcpyn",
    "CC_SHA1_Final",
    "vsnprintf",
    "_mbsnbset_s",
    "_i64tow",
    "SHA256_Init",
    "wvnsprintf",
    "RegCreateKey",
    "strtok_s",
    "_wctime32_s",
    "_i64toa",
    "CC_MD5_Final",
    "wmemcpy",
    "WinExec",
    "CreateDirectory*",
    "CC_SHA256_Update",
    "_vsnprintf_s_l",
    "jrand48",
    "wsprintf",
    "ldap_rename_ext_s",
    "filebuf.open",
    "_wsystem",
    "SHA256_Update",
    "_cwscanf_s",
    "wsprintfW",
    "_sntscanf",
    "_splitpath",
    "fscanf_s",
    "strpbrk",
    "wcstombs_s",
    "wscanf",
    "_mbsnbcat_s_l",
    "strcpynA",
    "pthread_cond_init",
    "wcsrtombs_s",
    "_wsopen_s",
    "CharToOemBuffA",
    "RIPEMD160_Update",
    "_tscanf",
    "HMAC",
    "StrCCpy",
    "Connection.connect",
    "lstrcatn",
    "_mbstok",
    "_mbsncpy",
    "CC_SHA384_Update",
    "create_directories",
    "pthread_mutex_unlock",
    "CFile.Open",
    "connect",
    "_vswprintf_s_l",
    "_snscanf_s_l",
    "fputc",
    "_wscanf_s",
    "_snprintf_s_l",
    "strtok",
    "_strtok_s_l",
    "lstrcatA",
    "snwscanf",
    "pthread_mutex_init",
    "fputs",
    "CC_SHA384_Init",
    "_putenv_s",
    "CharToOemBuffW",
    "pthread_mutex_trylock",
    "__wcstoul_internal",
    "_memccpy",
    "_snwprintf_s_l",
    "_strncpy*",
    "wmemset",
    "MD4_Init",
    "*RC4*",
    "strcpyW",
    "_ecvt_s",
    "memcpy_s",
    "erand48",
    "IsBadHugeReadPtr",
    "strcpyA",
    "HeapReAlloc",
    "memcpy",
    "ldap_rename_ext",
    "fopen_s",
    "srandom",
    "_cgetws_s",
    "_makepath",
    "SHA256_Final",
    "remove",
    "_mbsupr_s",
    "pthread_mutexattr_init",
    "__wcstold_internal",
    "StrCpy",
    "ldap_delete",
    "wmemmove_s",
    "_mkdir",
    "strcat",
    "_cscanf_s_l",
    "StrCAdd",
    "swprintf_s",
    "_strnset_s_l",
    "close",
    "ldap_delete_ext_s",
    "ldap_modrdn",
    "strchr",
    "_gmtime32_s",
    "_ftcscat",
    "lstrcatnA",
    "_tcsncat",
    "OemToChar",
    "mutex",
    "CharToOem",
    "strcpy_s",
    "lstrcatnW",
    "_wscanf_s_l",
    "__lxstat64",
    "memalign",
    "MD2_Init",
    "StrCatBuffW",
    "StrCpyN",
    "CC_MD5",
    "StrCpyA",
    "StrCatBuffA",
    "StrCpyW",
    "tmpnam_r",
    "_vsnprintf",
    "strcatA",
    "StrCpyNW",
    "_mbsnbset_s_l",
    "EVP_DigestInit",
    "_stscanf",
    "CC_MD2",
    "_tcscat",
    "StrCpyNA",
    "xmalloc",
    "_tcslen",
    "*MD4*",
    "vasprintf",
    "strxfrm",
    "chmod",
    "ldap_add_ext",
    "alloca",
    "_snscanf_s",
    "IsBadWritePtr",
    "swscanf_s",
    "wmemcpy_s",
    "_itoa",
    "_ui64toa_s",
    "EVP_DigestUpdate",
    "__wcstol_internal",
    "_itow",
    "StrNCatW",
    "strncat_s",
    "ualstrcpy",
    "execvp",
    "_mbccat",
    "EVP_MD_CTX_init",
    "assert",
    "ofstream.write",
    "ldap_add",
    "_sscanf_s_l",
    "drand48",
    "CharToOemW",
    "swscanf",
    "_itow_s",
    "RIPEMD160_Init",
    "CopyMemory",
    "initstate",
    "getpwuid",
    "vsprintf",
    "_fcvt_s",
    "CharToOemA",
    "setuid",
    "malloc",
    "StrCatNA",
    "strcat_s",
    "srand",
    "getwd",
    "_controlfp_s",
    "olestrcpy",
    "__wcstod_internal",
    "_mbsnbcat",
    "lstrncat",
    "des_*",
    "CC_SHA224_Init",
    "set*",
    "vsprintf_s",
    "SHA1_Final",
    "_umask_s",
    "gets",
    "setstate",
    "wvsprintfW",
    "LoadLibraryEx",
    "ofstream.open",
    "calloc",
    "_mbstrlen",
    "_cgets_s",
    "_sopen_s",
    "IsBadStringPtr",
    "wcsncat_s",
    "add*",
    "nrand48",
    "create_directory",
    "ldap_search_ext",
    "_i64toa_s",
    "_ltoa_s",
    "_cwscanf_s_l",
    "wmemcmp",
    "__lxstat",
    "lstrlen",
    "pthread_condattr_destroy",
    "_ftcscpy",
    "wcstok_s",
    "__xmknod",
    "pthread_attr_destroy",
    "sethostname",
    "_fscanf_s_l",
    "StrCatN",
    "RegEnumKey",
    "_tcsncpy",
    "strcatW",
    "AfxLoadLibrary",
    "setenv",
    "tmpnam",
    "_mbsncat_s_l",
    "_wstrdate_s",
    "_wctime64_s",
    "_i64tow_s",
    "CC_MD4_Update",
    "ldap_add_s",
    "_umask",
    "CC_SHA1_Update",
    "_wcsset_s_l",
    "_mbsupr_s_l",
    "strstr",
    "_tsplitpath",
    "memmove",
    "_tcscpy",
    "vsnprintf_s",
    "strcmp",
    "wvnsprintfW",
    "tmpfile",
    "ldap_modify",
    "_mbsncat*",
    "mrand48",
    "sizeof",
    "StrCatA",
    "_ltow_s",
    "*desencrypt*",
    "StrCatW",
    "_mbccpy",
    "CC_MD2_Init",
    "RIPEMD160",
    "ldap_search",
    "CC_SHA224",
    "mbsrtowcs_s",
    "update",
    "ldap_delete_s",
    "getnameinfo",
    "*RC5*",
    "_wcsncat_s_l",
    "DriverManager.getConnection",
    "socket",
    "_cscanf_s",
    "ldap_modrdn_s",
    "_wopen",
    "CC_SHA256_Final",
    "_snwprintf*",
    "MD2_Update",
    "strcpy",
    "_strncat_s_l",
    "CC_MD5_Init",
    "mbscpy",
    "wmemmove",
    "LoadLibraryW",
    "_mbslen",
    "*alloc",
    "_mbsncat_s",
    "LoadLibraryA",
    "fopen",
    "StrLen",
    "delete",
    "_splitpath_s",
    "CreateFileTransacted*",
    "MD4_Final",
    "_open",
    "CC_SHA384",
    "wcslen",
    "wcsncat",
    "_mktemp_s",
    "pthread_mutexattr_destroy",
    "_snwscanf_s",
    "_strset_s",
    "_wcsncpy_s_l",
    "CC_MD2_Final",
    "_mbstok_s_l",
    "wctomb_s",
    "MySQL_Driver.connect",
    "_snwscanf_s_l",
    "*_des_*",
    "LoadLibrary",
    "_swscanf_s_l",
    "ldap_compare_s",
    "ldap_compare_ext",
    "_strlwr_s",
    "GetEnvironmentVariable",
    "cuserid",
    "_mbscat_s",
    "strspn",
    "_mbsncpy_s",
    "ldap_modrdn2",
    "LeaveCriticalSection",
    "CopyFile",
    "getpwd",
    "sscanf",
    "creat",
    "RegSetValue",
    "ldap_modrdn2_s",
    "CFile.Close",
    "*SHA_1*",
    "pthread_cond_destroy",
    "CC_SHA512_Update",
    "*RC2*",
    "StrNCatA",
    "_mbsnbcpy",
    "_mbsnset_s",
    "crypt",
    "excel",
    "_vstprintf",
    "xstrdup",
    "wvsprintfA",
    "getopt",
    "mkstemp",
    "_wcsnset_s",
    "_stprintf",
    "_sntprintf",
    "tmpfile_s",
    "OpenDocumentFile",
    "_mbsset_s_l",
    "_strset_s_l",
    "_strlwr_s_l",
    "ifstream.open",
    "xcalloc",
    "StrNCpyA",
    "_wctime_s",
    "CC_SHA224_Update",
    "_ctime64_s",
    "MoveFile",
    "chown",
    "StrNCpyW",
    "IsBadReadPtr",
    "_ui64tow_s",
    "IsBadCodePtr",
    "getc",
    "OracleCommand.ExecuteOracleScalar",
    "AccessDataSource.Insert",
    "IDbDataAdapter.FillSchema",
    "IDbDataAdapter.Update",
    "GetWindowText*",
    "SendMessage",
    "SqlCommand.ExecuteNonQuery",
    "streambuf.sgetc",
    "streambuf.sgetn",
    "OracleCommand.ExecuteScalar",
    "SqlDataSource.Update",
    "_Read_s",
    "IDataAdapter.Fill",
    "_wgetenv",
    "_RecordsetPtr.Open*",
    "AccessDataSource.Delete",
    "Recordset.Open*",
    "filebuf.sbumpc",
    "DDX_*",
    "RegGetValue",
    "fstream.read*",
    "SqlCeCommand.ExecuteResultSet",
    "SqlCommand.ExecuteXmlReader",
    "main",
    "streambuf.sputbackc",
    "read",
    "m_lpCmdLine",
    "CRichEditCtrl.Get*",
    "istream.putback",
    "SqlCeCommand.ExecuteXmlReader",
    "SqlCeCommand.BeginExecuteXmlReader",
    "filebuf.sgetn",
    "OdbcDataAdapter.Update",
    "filebuf.sgetc",
    "SQLPutData",
    "recvfrom",
    "OleDbDataAdapter.FillSchema",
    "IDataAdapter.FillSchema",
    "CRichEditCtrl.GetLine",
    "DbDataAdapter.Update",
    "SqlCommand.ExecuteReader",
    "istream.get",
    "ReceiveFrom",
    "_main",
    "fgetc",
    "DbDataAdapter.FillSchema",
    "kbhit",
    "UpdateCommand.Execute*",
    "Statement.execute",
    "fgets",
    "SelectCommand.Execute*",
    "getch",
    "OdbcCommand.ExecuteNonQuery",
    "CDaoQueryDef.Execute",
    "fstream.getline",
    "ifstream.getline",
    "SqlDataAdapter.FillSchema",
    "OleDbCommand.ExecuteReader",
    "Statement.execute*",
    "SqlCeCommand.BeginExecuteNonQuery",
    "OdbcCommand.ExecuteScalar",
    "SqlCeDataAdapter.Update",
    "sendmessage",
    "mysqlpp.DBDriver",
    "fstream.peek",
    "Receive",
    "CDaoRecordset.Open",
    "OdbcDataAdapter.FillSchema",
    "_wgetenv_s",
    "OleDbDataAdapter.Update",
    "readsome",
    "SqlCommand.BeginExecuteXmlReader",
    "recv",
    "ifstream.peek",
    "_Main",
    "_tmain",
    "_Readsome_s",
    "SqlCeCommand.ExecuteReader",
    "OleDbCommand.ExecuteNonQuery",
    "fstream.get",
    "IDbCommand.ExecuteScalar",
    "filebuf.sputbackc",
    "IDataAdapter.Update",
    "streambuf.sbumpc",
    "InsertCommand.Execute*",
    "RegQueryValue",
    "IDbCommand.ExecuteReader",
    "SqlPipe.ExecuteAndSend",
    "Connection.Execute*",
    "getdlgtext",
    "ReceiveFromEx",
    "SqlDataAdapter.Update",
    "RegQueryValueEx",
    "SQLExecute",
    "pread",
    "SqlCommand.BeginExecuteReader",
    "AfxWinMain",
    "getchar",
    "istream.getline",
    "SqlCeDataAdapter.Fill",
    "OleDbDataReader.ExecuteReader",
    "SqlDataSource.Insert",
    "istream.peek",
    "SendMessageCallback",
    "ifstream.read*",
    "SqlDataSource.Select",
    "SqlCommand.ExecuteScalar",
    "SqlDataAdapter.Fill",
    "SqlCommand.BeginExecuteNonQuery",
    "getche",
    "SqlCeCommand.BeginExecuteReader",
    "getenv",
    "streambuf.snextc",
    "Command.Execute*",
    "_CommandPtr.Execute*",
    "SendNotifyMessage",
    "OdbcDataAdapter.Fill",
    "AccessDataSource.Update",
    "fscanf",
    "QSqlQuery.execBatch",
    "DbDataAdapter.Fill",
    "cin",
    "DeleteCommand.Execute*",
    "QSqlQuery.exec",
    "PostMessage",
    "ifstream.get",
    "filebuf.snextc",
    "IDbCommand.ExecuteNonQuery",
    "Winmain",
    "fread",
    "getpass",
    "GetDlgItemTextCCheckListBox.GetCheck",
    "DISP_PROPERTY_EX",
    "pread64",
    "Socket.Receive*",
    "SACommand.Execute*",
    "SQLExecDirect",
    "SqlCeDataAdapter.FillSchema",
    "DISP_FUNCTION",
    "OracleCommand.ExecuteNonQuery",
    "CEdit.GetLine",
    "OdbcCommand.ExecuteReader",
    "CEdit.Get*",
    "AccessDataSource.Select",
    "OracleCommand.ExecuteReader",
    "OCIStmtExecute",
    "getenv_s",
    "DB2Command.Execute*",
    "OracleDataAdapter.FillSchema",
    "OracleDataAdapter.Fill",
    "CComboBox.Get*",
    "SqlCeCommand.ExecuteNonQuery",
    "OracleCommand.ExecuteOracleNonQuery",
    "mysqlpp.Query",
    "istream.read*",
    "CListBox.GetText",
    "SqlCeCommand.ExecuteScalar",
    "ifstream.putback",
    "readlink",
    "CHtmlEditCtrl.GetDHtmlDocument",
    "PostThreadMessage",
    "CListCtrl.GetItemText",
    "OracleDataAdapter.Update",
    "OleDbCommand.ExecuteScalar",
    "stdin",
    "SqlDataSource.Delete",
    "OleDbDataAdapter.Fill",
    "fstream.putback",
    "IDbDataAdapter.Fill",
    "_wspawnl",
    "fwprintf",
    "sem_wait",
    "_unlink",
    "ldap_search_ext_sW",
    "signal",
    "PQclear",
    "PQfinish",
    "PQexec",
    "PQresultStatus",
])
