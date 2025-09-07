class LayerOutputHook:
    """注册hook，保存某层的forward输出"""
    def __init__(self, module):
        self.output = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.output = output

    def remove(self):
        self.hook.remove()

class NeckFeatureExtractor:
    """
    提取YOLOv8 Layer 4, 6, 9特征
    """
    def __init__(self, model):
        self.model = model
        # YOLOv8主干模型 Sequential
        seq = self.model.model
        # 只关注4/6/9层
        self.target_indices = [4, 6, 9]
        self.hooks = []
        self.outputs = [None, None, None]
        for idx in self.target_indices:
            hook = LayerOutputHook(seq[idx])
            self.hooks.append(hook)

    def clear(self):
        # 钩子自动覆盖，无需clear
        pass

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_features(self):
        # 钩子按顺序保存4/6/9层输出
        feats = [h.output for h in self.hooks]
        # 检查是否都拿到
        if all(f is not None for f in feats):
            return feats
        else:
            return None