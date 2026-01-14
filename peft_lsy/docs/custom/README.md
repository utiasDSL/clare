#### LoRA Design
```
peft_model: PeftModel
    base_model: LoraModel(BaseTuner)
        model: Transformers
            linear: lora.Linear(nn.Module, BaseTunerLayer)
                base_layer: Linear
                lora_a: ModuleDict
                    lora_a_en: Linear
                    lora_a_de: Linear
                lora_b: ModuleDict
                    lora_b_en: Linear
                    lora_b_de: Linear
```

#### Ours Design
```
peft_model: PeftModel
    base_model: OurAdapterModel(BaseTuner)
        model: nn.Module
            ...
                mlp: OurAdapterLayer(nn.Module, BaseTunerLayer)
                    base_layer: nn.Module
                    layer_name: str
                    layer_id: int

                    our_adapters: nn.ModuleList
                    [   
                        adapter: OurAdapter
                            discriminators: nn.ModuleList
                            [
                                discriminator: Discriminator
                            ] x N
                            func_adapter: FuncAdapter
                    ] x N

                    _discriminators: List[Discriminator] # a short cut to access all discriminators
                    _func_adapters: List[FuncAdapter]  # a short cut to access all func adapters
                    _mapping: List
        _adapted_layers: OrderedDict[str, OurAdapterLayer] # a short cut to access all adapted layers

```