![JetInfero](media/jetinfero.png)  
[![Chat on Discord](https://img.shields.io/discord/754884471324672040?style=for-the-badge)](https://discord.gg/tPWjMwK)
[![Follow on Bluesky](https://img.shields.io/badge/Bluesky-tinyBigGAMES-blue?style=for-the-badge&logo=bluesky)](https://bsky.app/profile/tinybiggames.com)

## 🌟 Fast, Flexible Local LLM Inference for Developers 🚀

JetInfero is a nimble and high-performance library that enables developers to integrate local Large Language Models (LLMs) effortlessly into their applications. Powered by **llama.cpp** 🕊️, JetInfero prioritizes speed, flexibility, and ease of use 🌐. It’s compatible with any language supporting **Win64**, **Unicode**, and dynamic-link libraries (DLLs).

## 💡 Why Choose JetInfero?

- **Optimized for Speed** ⚡️: Built on llama.cpp, JetInfero offers lightning-fast inference capabilities with minimal overhead.
- **Cross-Language Support** 🌐: Seamlessly integrates with Delphi, C++, C#, Java, and other Win64-compatible environments.
- **Intuitive API** 🔬: A clean procedural API simplifies model management, inference execution, and callback handling.
- **Customizable Templates** 🖋️: Tailor input prompts to suit different use cases with ease.
- **Scalable Performance** 🚀: Leverage GPU acceleration, token streaming, and multi-threaded execution for demanding workloads.

## 🛠️ Key Features

### 🤖 Advanced AI Integration

JetInfero expands your toolkit with capabilities such as:

- Dynamic chatbot creation 🗣️.
- Automated text generation 🔄 and summarization 🕻.
- Context-aware content creation 🌐.
- Real-time token streaming for adaptive applications ⌚.

### 🔒 Privacy-Centric Local Execution

- Operates entirely offline 🔐, ensuring sensitive data remains secure.
- GPU acceleration supported via Vulkan for enhanced performance 🚒.

### ⚙️ Performance Optimization

- Configure GPU utilization with `AGPULayers` 🔄.
- Allocate threads dynamically using `AMaxThreads` 🌐.
- Access performance metrics to monitor throughput and efficiency 📊.

### 🔀 Flexible Prompt Templates

JetInfero’s template system simplifies input customization. Templates include placeholders such as:

- **`{role}`**: Denotes the sender’s role (e.g., `user`, `assistant`).
- **`{content}`**: Represents the message content.

For example:

```pascal
  jiDefineModel(
    // Model Filename
    'C:/LLM/GGUF/hermes-3-llama-3.2-3b-abliterated-q8_0.gguf', 
    
    // Model Refname
    'hermes-3-llama-3.2-3b-abliterated-q8_0',                  
    
     // Model Template
    '<|im_start|>system\n{content}<|im_end|>',                
    
     // Model Template End
    '<|im_start|>assistant\n',                                
    
    // Capitalize Role
    False,                                                     
    
    // Max Context
    8192,                                                      
    
    // Main GPU, -1 for best, 0..N GPU number
    -1,                                                        
    
    // GPU Layers, -1 for max, 0 for CPU only, 1..N for layer
    -1,                                                        
    
    // Max threads, default 4, max will be physical CPU count     
     4                                                         
  );
```

#### Template Benefits

- **Adaptability** 🌐: Customize prompts for various LLMs and use cases.
- **Consistency** 🔄: Ensure predictable inputs for reliable results.
- **Flexibility** 🌈: Modify prompt formats for tasks like JSON or markdown generation.

### 🍂 Streamlined Model Management

- Define models with `jiDefineModel` 🔨.
- Load/unload models dynamically using `jiLoadModel` and `jiUnloadModel` 🔀.
- Save/load model configurations with `jiSaveModelDefines` and `jiLoadModelDefines` 🗃️.
- Clear all model definitions using `jiClearModelDefines` 🧹.

### 🔁 Inference Execution

- Perform inference tasks with `jiRunInference` ⚙️.
- Stream real-time tokens via `InferenceTokenCallback` ⌚.
- Retrieve responses using `jiGetInferenceResponse` 🖊️.

### 📊 Performance Monitoring

- Retrieve detailed metrics like tokens/second, input/output token counts, and execution time via `jiGetPerformanceResult` 📊.

## 🛠️ Installation

1. **Download the Repository** 📦
   - [Download here](https://github.com/tinyBigGAMES/JetInfero/archive/refs/heads/main.zip) and extract the files to your preferred directory 📂.

   Ensure `JetInfero.dll` is accessible in your project directory.

2. **Acquire a GGUF Model** 🧠
   - Obtain a model from [Hugging Face](https://huggingface.co), such as [
Hermes-3-Llama-3.2-3B-Q8_0-GGUF](https://huggingface.co/tinybiggames/Hermes-3-Llama-3.2-3B-Q8_0-GGUF/resolve/main/hermes-3-llama-3.2-3b-q8_0.gguf?download=true). Save it to a directory accessible to your application (e.g., `C:/LLM/GGUF`) 💾.

2. **Add JetInfero to Your Project** 🔨
   - Include the `JetInfero` unit in your Delphi project.   

4. **Ensure GPU Compatibility** 🎮
   - Verify Vulkan compatibility for enhanced performance ⚡. Adjust `AGPULayers` as needed to accommodate VRAM limitations 📉.
   
5. **Building JetInfero DLL** 🛠️  
   - Open and compile the `JetInfero.dproj` project 📂. This process will generate the 64-bit `JetInfero.dll` in the `lib` folder 🗂️.  
   - The project was created and tested using Delphi 12.2 on Windows 11 24H2 🖥️.  

6. **Using JetInfero** 🚀  
   - JetInfero can be used with any programming language that supports Win64 and Unicode bindings 💻.  
   - Ensure the `JetInfero.dll` is included in your distribution and accessible at runtime 📦.  

## 📈 Quick Start

### ⚙️ Basic Setup

Integrate JetInfero into your Delphi project:

```pascal
uses
  JetInfero;

var
  LTokensPerSec: Double;
  LTotalInputTokens: Int32;
  LTotalOutputTokens: Int32;
begin
  if jiInit() then
  begin
    jiDefineModel(
      'C:/LLM/GGUF/hermes-3-llama-3.2-3b-abliterated-q8_0.gguf',
      'hermes-3-llama-3.2-3b-abliterated-q8_0',
      '<|im_start|>system\n{content}<|im_end|>',
      '<|im_start|>assistant\n', False, 8192, -1, -1, 4);
    
    jiLoadModel('hermes-3-llama-3.2-3b-abliterated-q8_0');

    jiAddMessage('user', 'What is AI?');

    if jiRunInference(PWideChar(LModelRef)) then
      begin
        jiGetPerformanceResult(@LTokensPerSec, @LTotalInputTokens, @LTotalOutputTokens);
        WriteLn('Input Tokens : ', LTotalInputTokens);
        WriteLn('Output Tokens: ', LTotalOutputTokens);
        WriteLn('Speed        : ', LTokensPerSec:3:2, ' t/s');
      end
    else
      begin
        WriteLn('Error: ', jiGetLastError());
      end;

    jiUnloadModel();
    jiQuit();
  end;

end.
```

### 🔁 Using Callbacks

Define a custom callback to handle token streaming:

```pascal
procedure InferenceCallback(const Token: string; const UserData: Pointer);
begin
  Write(Token);
end;

jiSetTokenCallback(@InferenceCallback, nil);
```

### 📊 Retrieve Performance Metrics

Access performance results to monitor efficiency:

```pascal
var
  Metrics: TPerformanceResult;
begin
  Metrics := jiGetPerformanceResult();
  WriteLn('Tokens/Sec: ', Metrics.TokensPerSecond);
  WriteLn('Input Tokens: ', Metrics.TotalInputTokens);
  WriteLn('Output Tokens: ', Metrics.TotalOutputTokens);
end;
```

### 🛠️ Support and Resources

- Report issues via the [Issue Tracker](https://github.com/tinyBigGAMES/jetinfero/issues) 🐞.
- Engage in discussions on the [Forum](https://github.com/tinyBigGAMES/jetinfero/discussions) and [Discord](https://discord.gg/tPWjMwK) 💬.
- Learn more at [Learn Delphi](https://learndelphi.org) 📚.

### 🤝 Contributing  

Contributions to **✨ JetInfero** are highly encouraged! 🌟  
- 🐛 **Report Issues:** Submit issues if you encounter bugs or need help.  
- 💡 **Suggest Features:** Share your ideas to make **Lumina** even better.  
- 🔧 **Create Pull Requests:** Help expand the capabilities and robustness of the library.  

Your contributions make a difference! 🙌✨

#### Contributors 👥🤝
<br/>

<a href="https://github.com/tinyBigGAMES/JetInfero/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=tinyBigGAMES/JetInfero&max=500&columns=20&anon=1" />
</a>

### 📜 Licensing

**JetInfero** is distributed under the 🆓 **BSD-3-Clause License**, allowing for redistribution and use in both source and binary forms, with or without modification, under specific conditions. See the [LICENSE](https://github.com/tinyBigGAMES/JetInfero?tab=BSD-3-Clause-1-ov-file#BSD-3-Clause-1-ov-file) file for more details.

---
**Elevate your Delphi projects with JetInfero 🚀 – your bridge to seamless local generative AI integration 🤖.**  

<p align="center">
<img src="media/delphi.png" alt="Delphi">
</p>
<h5 align="center">

Made with :heart: in Delphi
