class LayerOutputHook:
    def __init__(self, module):
        self.output = None
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.output = output
    def remove(self):
        self.hook.remove()

class NeckFeatureExtractor:
    """
    专门提取YOLOv8 backbone到neck的三个尺度输出（Layer 4, 6, 9）
    """
    def __init__(self, model):
        self.model = model
        seq = self.model.model
        self.target_indices = [4, 6, 9]
        self.hooks = [LayerOutputHook(seq[i]) for i in self.target_indices]

    def clear(self):
        # 钩子只保存最新一轮forward的结果
        for h in self.hooks:
            h.output = None

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_features(self):
        feats = [h.output for h in self.hooks]
        if all(f is not None for f in feats):
            return feats
        else:
            return None