# Copyright (c) Alibaba, Inc. and its affiliates.
import enum


class Fields(object):
    """ Names for different application fields
    """
    cv = 'cv'
    nlp = 'nlp'
    audio = 'audio'
    multi_modal = 'multi-modal'
    science = 'science'

class MultiModalTasks(object):
    # multi-modal tasks
    image_captioning = 'image-captioning'
    visual_grounding = 'visual-grounding'
    text_to_image_synthesis = 'text-to-image-synthesis'
    multi_modal_embedding = 'multi-modal-embedding'
    generative_multi_modal_embedding = 'generative-multi-modal-embedding'
    multi_modal_similarity = 'multi-modal-similarity'
    visual_question_answering = 'visual-question-answering'
    visual_entailment = 'visual-entailment'
    video_multi_modal_embedding = 'video-multi-modal-embedding'
    image_text_retrieval = 'image-text-retrieval'
    document_vl_embedding = 'document-vl-embedding'
    video_captioning = 'video-captioning'
    video_question_answering = 'video-question-answering'
    video_temporal_grounding = 'video-temporal-grounding'
    text_to_video_synthesis = 'text-to-video-synthesis'
    efficient_diffusion_tuning = 'efficient-diffusion-tuning'


class TasksIODescriptions(object):
    image_to_image = 'image_to_image',
    images_to_image = 'images_to_image',
    image_to_text = 'image_to_text',
    seed_to_image = 'seed_to_image',
    text_to_speech = 'text_to_speech',
    text_to_text = 'text_to_text',
    speech_to_text = 'speech_to_text',
    speech_to_speech = 'speech_to_speech'
    speeches_to_speech = 'speeches_to_speech',
    visual_grounding = 'visual_grounding',
    visual_question_answering = 'visual_question_answering',
    visual_entailment = 'visual_entailment',
    generative_multi_modal_embedding = 'generative_multi_modal_embedding'
    efficient_diffusion_tuning = 'efficient_diffusion_tuning'


class Tasks(MultiModalTasks):
    reverse_field_index = {}

    @staticmethod
    def find_field_by_task(task_name):
        if len(Tasks.reverse_field_index) == 0:
            # Lazy init, not thread safe
            field_dict = {
                Fields.multi_modal: [
                    getattr(Tasks, attr) for attr in dir(MultiModalTasks)
                    if not attr.startswith('__')
                ]
            }

            for field, tasks in field_dict.items():
                for task in tasks:
                    if task in Tasks.reverse_field_index:
                        raise ValueError(f'Duplicate task: {task}')
                    Tasks.reverse_field_index[task] = field

        return Tasks.reverse_field_index.get(task_name)


class InputFields(object):
    """ Names for input data fields in the input data for pipelines
    """
    img = 'img'
    text = 'text'
    audio = 'audio'

class DatasetFormations(enum.Enum):
    """ How a dataset is organized and interpreted
    """
    # formation that is compatible with official huggingface dataset, which
    # organizes whole dataset into one single (zip) file.
    hf_compatible = 1
    # native modelscope formation that supports, among other things,
    # multiple files in a dataset
    native = 2
    # for local meta cache mark
    formation_mark_ext = '.formation_mark'


DatasetMetaFormats = {
    DatasetFormations.native: ['.json'],
    DatasetFormations.hf_compatible: ['.py'],
}


class ModelFile(object):
    CONFIGURATION = 'configuration.json'
    README = 'README.md'
    TF_SAVED_MODEL_FILE = 'saved_model.pb'
    TF_GRAPH_FILE = 'tf_graph.pb'
    TF_CHECKPOINT_FOLDER = 'tf_ckpts'
    TF_CKPT_PREFIX = 'ckpt-'
    TORCH_MODEL_FILE = 'pytorch_model.pt'
    TORCH_MODEL_BIN_FILE = 'pytorch_model.bin'
    VOCAB_FILE = 'vocab.txt'
    ONNX_MODEL_FILE = 'model.onnx'
    LABEL_MAPPING = 'label_mapping.json'
    TRAIN_OUTPUT_DIR = 'output'
    TRAIN_BEST_OUTPUT_DIR = 'output_best'
    TS_MODEL_FILE = 'model.ts'
    YAML_FILE = 'model.yaml'
    TOKENIZER_FOLDER = 'tokenizer'
    CONFIG = 'config.json'


class Invoke(object):
    KEY = 'invoked_by'
    PRETRAINED = 'from_pretrained'
    PIPELINE = 'pipeline'
    TRAINER = 'trainer'
    LOCAL_TRAINER = 'local_trainer'
    PREPROCESSOR = 'preprocessor'


class ThirdParty(object):
    KEY = 'third_party'
    EASYCV = 'easycv'
    ADASEQ = 'adaseq'
    ADADET = 'adadet'


class ConfigFields(object):
    """ First level keyword in configuration file
    """
    framework = 'framework'
    task = 'task'
    pipeline = 'pipeline'
    model = 'model'
    dataset = 'dataset'
    preprocessor = 'preprocessor'
    train = 'train'
    evaluation = 'evaluation'
    postprocessor = 'postprocessor'


class ConfigKeys(object):
    """Fixed keywords in configuration file"""
    train = 'train'
    val = 'val'
    test = 'test'


class Requirements(object):
    """Requirement names for each module
    """
    protobuf = 'protobuf'
    sentencepiece = 'sentencepiece'
    sklearn = 'sklearn'
    scipy = 'scipy'
    timm = 'timm'
    tokenizers = 'tokenizers'
    tf = 'tf'
    torch = 'torch'


class Frameworks(object):
    tf = 'tensorflow'
    torch = 'pytorch'
    kaldi = 'kaldi'


DEFAULT_MODEL_REVISION = None
MASTER_MODEL_BRANCH = 'master'
DEFAULT_REPOSITORY_REVISION = 'master'
DEFAULT_DATASET_REVISION = 'master'
DEFAULT_DATASET_NAMESPACE = 'modelscope'
DEFAULT_DATA_ACCELERATION_ENDPOINT = 'https://oss-accelerate.aliyuncs.com'


class ModeKeys:
    TRAIN = 'train'
    EVAL = 'eval'
    INFERENCE = 'inference'


class LogKeys:
    ITER = 'iter'
    ITER_TIME = 'iter_time'
    EPOCH = 'epoch'
    LR = 'lr'  # learning rate
    MODE = 'mode'
    DATA_LOAD_TIME = 'data_load_time'
    ETA = 'eta'  # estimated time of arrival
    MEMORY = 'memory'
    LOSS = 'loss'


class TrainerStages:
    after_init = 'after_init'
    before_run = 'before_run'
    before_val = 'before_val'
    before_train_epoch = 'before_train_epoch'
    before_train_iter = 'before_train_iter'
    after_train_iter = 'after_train_iter'
    after_train_epoch = 'after_train_epoch'
    before_val_epoch = 'before_val_epoch'
    before_val_iter = 'before_val_iter'
    after_val_iter = 'after_val_iter'
    after_val_epoch = 'after_val_epoch'
    after_run = 'after_run'
    after_val = 'after_val'


class ColorCodes:
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    END = '\033[0m'


class Devices:
    """device used for training and inference"""
    cpu = 'cpu'
    gpu = 'gpu'


# Supported extensions for text datasets.
EXTENSIONS_TO_LOAD = {
    'csv': 'csv',
    'tsv': 'csv',
    'json': 'json',
    'jsonl': 'json',
    'parquet': 'parquet',
    'txt': 'text'
}


class DatasetPathName:
    META_NAME = 'meta'
    DATA_FILES_NAME = 'data_files'
    LOCK_FILE_NAME_ANY = 'any'
    LOCK_FILE_NAME_DELIMITER = '-'


class MetaDataFields:
    ARGS_BIG_DATA = 'big_data'


DatasetVisibilityMap = {1: 'private', 3: 'internal', 5: 'public'}


class DistributedParallelType(object):
    """Parallel Strategies for Distributed Models"""
    DP = 'data_parallel'
    TP = 'tensor_model_parallel'
    PP = 'pipeline_model_parallel'


class DatasetTensorflowConfig:
    BATCH_SIZE = 'batch_size'
    DEFAULT_BATCH_SIZE_VALUE = 5