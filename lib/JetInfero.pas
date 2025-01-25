{===============================================================================
     _       _    ___         __
  _ | | ___ | |_ |_ _| _ _   / _| ___  _ _  ___ ™
 | || |/ -_)|  _| | | | ' \ |  _|/ -_)| '_|/ _ \
  \__/ \___| \__||___||_||_||_|  \___||_|  \___/
            Local LLM Inference Library

 Copyright © 2024-present tinyBigGAMES™ LLC
 All Rights Reserved.

 https://github.com/tinyBigGAMES/JetInfero

BSD 3-Clause License

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.

-------------------------------------------------------------------------------

 This project uses the following open-source libraries:
 * llama.cpp - https://github.com/ggerganov/llama.cpp

-------------------------------------------------------------------------------

 >>> USAGE NOTES <<<
 ===================

 - You can download GGUF models from https://huggingface.co, such as
   https://huggingface.co/bartowski/gemma-2-2b-it-abliterated-GGUF, which is
   a fast, light models referenced in the examples (Q8 quantized version).

 - You can also convert a model to GGUF format using an online converter:
   https://huggingface.co/spaces/ggml-org/gguf-my-repo. You will have have
   a HF account as it will save the converted model to your own account.

 - Setting in jiDefineModel, setting MainGPU to -1 will try to select the best
   GPU on your system. Otherwise setting it to 0 - N will try to use the
   specified GPU. Setting MaxGPULayers to -1 will use try to use all
   available layers on the GPU. Setting it to 0 will use the CPU only, 1 - N,
   will offload on to specified number of GPULayers.

 - You can set up the variouse callbacks to control model output.

 - JetInfero is optimized for local inference on consumer hardware, as such
   using a smaller model, typically 4 bit quantized at around 1-3 GB in size
   will load and run fast on modern consumer grade GPUs.

 1. Downloading GGUF Models:
    - You can download GGUF models from [Hugging Face](https://huggingface.co),
      such as gemma-2-2b-it-abliterated-GGUF
      (https://huggingface.co/bartowski/gemma-2-2b-it-abliterated-GGUF).
      This model is referenced in examples and is available in a fast,
      lightweight Q8-quantized version.

 2. Converting Models to GGUF Format:
    - You can convert a model to the GGUF format using an online converter
      available at Hugging Face Spaces
      (https://huggingface.co/spaces/ggml-org/gguf-my-repo).
      Note that you need a Hugging Face account, as the converted model will
      be saved to your account.

 3. GPU Settings in `jiDefineModel`:
    - Setting `MainGPU` to `-1` will automatically select the best GPU
      available on your system.
    - Alternatively, you can specify a GPU by setting `MainGPU` to `0 - N`
      (where `N` is the GPU index).
    - For `MaxGPULayers`:
      - Setting it to `-1` will use all available layers on the GPU.
      - Setting it to `0` will use the CPU only.
      - Setting it to `1 - N` will offload a specific number of layers to the
        GPU.

 4. Customizing Output:
    - You can configure various callbacks to control the model's output
      according to your needs.

 5. Optimized for Local Inference:
    - JetInfero is designed for efficient local inference on consumer-grade
      hardware. Using smaller models (typically 4-bit quantized, around 1–3
      GB in size) ensures fast loading and performance on modern consumer GPUs.

-------------------------------------------------------------------------------

 >>> CHANGELOG <<<
 =================

 Version 0.1.0:
  - Initial release
===============================================================================}

/// <summary>
///   The <c>JetInfero</c> unit provides an interface for managing and performing inference tasks
///   with local Large Language Models (LLMs). This unit includes functions and procedures
///   for model management, configuration, inference execution, and performance monitoring.
/// </summary>
/// <remarks>
///   This unit serves as the primary API for interacting with the JetInfero library, enabling developers
///   to integrate LLM capabilities into their Delphi applications. Key features include:
///
///   <para><b>Model Management:</para></b>
///   - Define and configure models using <c>jiDefineModel</c>.
///   - Load and unload models dynamically using <c>jiLoadModel</c> and <c>jiUnloadModel</c>.
///   - Save and load model definitions to/from JSON files for external editing with <c>jiSaveModelDefines</c> and <c>jiLoadModelDefines</c>.
///   - Clear all defined models with <c>jiClearModelDefines</c>.
///
///   <para><b>Message Management:</para></b>
///   - Add conversation context using <c>jiAddMessage</c>.
///   - Clear all messages with <c>jiClearMessages</c>.
///   - Retrieve the last user message with <c>jiGetLastUserMessage</c>.
///   - Construct prompts from the message history for inference using <c>jiGetMessagePrompt</c>.
///
///   <para><b>Inference Execution:</para></b>
///   - Run inference tasks with <c>jiRunInference</c>, supporting real-time token streaming via callbacks.
///   - Retrieve generated responses with <c>jiGetInferenceResponse</c>.
///
///   <para><b>Performance Monitoring:</para></b>
///   - Obtain detailed metrics, including tokens per second and input/output token counts, using <c>jiGetPerformanceResult</c>.
///
///   <para><b>Configuration Management:</para></b>
///   - Save and load library configurations using <c>jiSaveConfig</c> and <c>jiLoadConfig</c>.
///
///   <para><b>Callbacks:</para></b>
///   - Define custom callbacks for handling inference tokens, progress updates, model load events, and informational messages.
///   - Use <c>jiSetInferenceTokenCallback</c>, <c>jiSetLoadModelProgressCallback</c>, <c>jiSetInfoCallback</c>, and related functions.
///
///   <para><b>Error Handling:</para></b>
///   - Retrieve error messages from the library using <c>jiGetLastError</c>.
///
///   <para><b>Usage Notes:</para></b>
///   - Ensure the JetInfero library is properly initialized using <c>jiInit</c> before invoking other functions.
///   - Call <c>jiQuit</c> to clean up resources when the library is no longer needed.
///   - Use GPU acceleration for models where supported to enhance performance.
/// </remarks>
/// <example>
///   Below is an example of a typical workflow using the <c>JetInfero</c> unit:
///   <code>
///   if jiInit() then
///   begin
///     jiDefineModel('model.json', 'myModel', '{prompt}', '{end}', False, 2048, 0, -1, 8);
///     jiLoadModel('myModel');
///     jiAddMessage('user', 'Hello, JetInfero!');
///     if jiRunInference('myModel') then
///       WriteLn(jiGetInferenceResponse());
///     jiUnloadModel();
///     jiQuit();
///   end
///   else
///     WriteLn(jiGetLastError());
///   </code>
/// </example>
unit JetInfero;

{$IFDEF FPC}
{$MODE DELPHIUNICODE}
{$ENDIF}

{$IFNDEF WIN64}
  // Generates a compile-time error if the target platform is not Win64
  {$MESSAGE Error 'Unsupported platform'}
{$ENDIF}

{$Z4}  // Sets the enumeration size to 4 bytes
{$A8}  // Sets the alignment for record fields to 8 bytes

interface

type
  PInt32 = ^Int32;

const

  /// <summary>
  ///   The name of the JetInfero dynamic-link library (DLL).
  /// </summary>
  /// <remarks>
  ///   This library contains the core functionality required for the JetInfero application.
  ///   It is dynamically linked at runtime, enabling modular and reusable components.
  /// </remarks>
  JETINFERO_DLL = 'JetInfero.dll';

  /// <summary>
  ///   Role identifier for the system in the LLM messaging system.
  /// </summary>
  /// <remarks>
  ///   The system role represents a special entity in the messaging framework that provides
  ///   overarching instructions or context to guide the behavior of the assistant. Unlike
  ///   the user or assistant roles, the system role typically defines rules, goals, or
  ///   parameters for the conversation.
  ///
  ///   <para><b>Use Cases:</b></para>
  ///   <para>- Setting conversation guidelines (e.g., tone, format, or context) for the assistant.</para>
  ///   <para>- Providing high-level instructions for how the assistant should respond.</para>
  ///   <para>- Configuring defaults for the assistant's behavior in a particular interaction.</para>
  ///
  ///   <para><b>Examples:</b></para>
  ///   <para>- A system message might specify, "You are a helpful assistant that provides concise answers."</para>
  ///   <para>- Alternatively, it might define output format requirements, such as "Respond in JSON format only."</para>
  ///
  ///   <para><b>Notes:</b></para>
  ///   <para>- System messages are generally added at the start of the conversation to establish context.</para>
  ///   <para>- They are not directly visible to end-users in typical scenarios but are integral to the assistant's behavior.</para>
  /// </remarks>
  jiROLE_SYSTEM = 'system';

  /// <summary>
  ///   Role identifier for a user in the LLM messaging system.
  /// </summary>
  /// <remarks>
  ///   The user represents the primary entity interacting with the system. This role is assigned
  ///   to the individual or application initiating queries, tasks, or messages to the LLM.
  ///   Typical actions for the user role include asking questions, providing input, or issuing commands.
  /// </remarks>
  jiROLE_USER      = 'user';

  /// <summary>
  ///   Role identifier for the assistant in the LLM messaging system.
  /// </summary>
  /// <remarks>
  ///   The assistant role represents the LLM itself or the chatbot that processes user input
  ///   and generates responses. This role encapsulates the logic, reasoning, and language
  ///   generation capabilities provided by the LLM.
  /// </remarks>
  jiROLE_ASSISTANT = 'assistant';

  /// <summary>
  ///   Role identifier for a tool in the LLM messaging system.
  /// </summary>
  /// <remarks>
  ///   Tools refer to external systems, APIs, or utilities that augment the assistant's capabilities.
  ///   They are often invoked by the assistant to perform specific tasks, such as retrieving data,
  ///   processing complex computations, or handling specialized queries.
  ///   The tool role facilitates extensibility and ensures the LLM can integrate seamlessly with other systems.
  /// </remarks>
  jiROLE_TOOL      = 'tool';

type
  /// <summary>
  ///   Callback function to determine whether an inference process should be canceled.
  /// </summary>
  /// <param name="AUserData">
  ///   Custom user data passed to the callback. This can be used to track context or associate with specific operations.
  /// </param>
  /// <returns>
  ///   Returns <c>True</c> to indicate that the inference should be canceled; otherwise, <c>False</c>.
  /// </returns>
  jiInferenceCancelCallback = function(const AUserData: Pointer): Boolean; cdecl;

  /// <summary>
  ///   Callback procedure invoked when a token is generated during the inference process.
  /// </summary>
  /// <param name="AToken">
  ///   The token generated during the inference process.
  /// </param>
  /// <param name="AUserData">
  ///   Custom user data passed to the callback. This can be used for additional context or tracking.
  /// </param>
  /// <remarks>
  ///   Use this callback to handle generated tokens, such as streaming output to a user interface or processing
  ///   them in real time.
  /// </remarks>
  jiInferenceTokenCallback = procedure(const AToken: PWideChar; const AUserData: Pointer); cdecl;

  /// <summary>
  ///   Callback procedure for logging informational messages from the library.
  /// </summary>
  /// <param name="ALevel">
  ///   The severity or log level of the message. Common levels might include info, warning, or error.
  /// </param>
  /// <param name="AText">
  ///   The text message to log.
  /// </param>
  /// <param name="AUserData">
  ///   Custom user data passed to the callback. This can provide additional context for logging operations.
  /// </param>
  /// <remarks>
  ///   Use this callback to integrate library logs with your application’s logging or debugging system.
  /// </remarks>
  jiInfoCallback = procedure(const ALevel: Integer; const AText: PWideChar; const AUserData: Pointer); cdecl;

  /// <summary>
  ///   Callback function invoked to report the progress of loading a model.
  /// </summary>
  /// <param name="AModelName">
  ///   The name of the model being loaded.
  /// </param>
  /// <param name="AProgress">
  ///   The progress percentage (0.0 to 100.0) of the model loading process.
  /// </param>
  /// <param name="AUserData">
  ///   Custom user data passed to the callback.
  /// </param>
  /// <returns>
  ///   Returns <c>True</c> to continue loading the model; <c>False</c> to cancel.
  /// </returns>
  /// <remarks>
  ///   Use this callback to display progress bars or track model loading status in your application.
  /// </remarks>
  jiLoadModelProgressCallback = function(const AModelName: PWideChar; const AProgress: Single; const AUserData: Pointer): Boolean; cdecl;

  /// <summary>
  ///   Callback procedure invoked upon completion of a model loading process.
  /// </summary>
  /// <param name="AModelName">
  ///   The name of the model that was loaded.
  /// </param>
  /// <param name="ASuccess">
  ///   Indicates whether the model was successfully loaded (<c>True</c>) or if an error occurred (<c>False</c>).
  /// </param>
  /// <param name="AUserData">
  ///   Custom user data passed to the callback.
  /// </param>
  /// <remarks>
  ///   Use this callback to respond to the completion of a model loading operation, such as logging the result
  ///   or updating the user interface.
  /// </remarks>
  jiLoadModelCallback = procedure(const AModelName: PWideChar; const ASuccess: Boolean; const AUserData: Pointer); cdecl;

  /// <summary>
  ///   Callback procedure invoked at the start of an inference process.
  /// </summary>
  /// <param name="AUserData">
  ///   Custom user data passed to the callback.
  /// </param>
  /// <remarks>
  ///   Use this callback to perform any setup or initialization required before the inference begins.
  /// </remarks>
  jiInferenceStartCallback = procedure(const AUserData: Pointer); cdecl;

  /// <summary>
  ///   Callback procedure invoked at the end of an inference process.
  /// </summary>
  /// <param name="AUserData">
  ///   Custom user data passed to the callback.
  /// </param>
  /// <remarks>
  ///   Use this callback to perform any cleanup or post-processing tasks after the inference has completed.
  /// </remarks>
  jiInferenceEndCallback = procedure(const AUserData: Pointer); cdecl;

/// <summary>
///   Initializes the JetInfero library for use.
/// </summary>
/// <returns>
///   Returns <c>True</c> if the library was successfully initialized; otherwise, <c>False</c>.
/// </returns>
/// <remarks>
///   This function must be called before using any other functionality of the JetInfero library.
///   It performs necessary setup and prepares the library for operation.
///   If the function returns <c>False</c>, use <c>jiGetLastError</c> to retrieve additional error information.
/// </remarks>
function jiInit(): Boolean; cdecl; external JETINFERO_DLL;

/// <summary>
///   Checks if the JetInfero library has already been initialized.
/// </summary>
/// <returns>
///   Returns <c>True</c> if the library has been initialized; otherwise, <c>False</c>.
/// </returns>
/// <remarks>
///   Use this function to verify that <c>jiInit</c> has been successfully called before proceeding
///   with operations that require the library to be initialized.
/// </remarks>
function jiWasInit(): Boolean; cdecl; external JETINFERO_DLL;

/// <summary>
///   Retrieves the current version of the JetInfero library.
/// </summary>
/// <returns>
///   Returns a wide character pointer (<c>PWideChar</c>) to a string containing the version number of the library.
/// </returns>
/// <remarks>
///   The version information can be used to ensure compatibility between the application and the library.
///   The returned string remains valid for the lifetime of the library session.
/// </remarks>
function jiGetVersion(): PWideChar; cdecl; external JETINFERO_DLL;

/// <summary>
///   Shuts down and releases resources used by the JetInfero library.
/// </summary>
/// <remarks>
///   This function should be called when the application is done using the JetInfero library to ensure
///   that all allocated resources are properly freed and no memory leaks occur.
///   After calling this function, the library must be reinitialized using <c>jiInit</c> before further use.
/// </remarks>
procedure jiQuit(); cdecl; external JETINFERO_DLL;

/// <summary>
///   Retrieves the last error message from the JetInfero library.
/// </summary>
/// <returns>
///   Returns a wide character pointer (<c>PWideChar</c>) to a string containing the last error message.
/// </returns>
/// <remarks>
///   This function provides additional context for debugging when a library function fails or behaves unexpectedly.
///   The returned error message is specific to the last function call that encountered an error.
///   The message is valid until the next function call or until the library is unloaded.
/// </remarks>
function jiGetLastError(): PWideChar; cdecl; external JETINFERO_DLL;

/// <summary>
///   Retrieves the current inference cancellation callback function.
/// </summary>
/// <returns>
///   Returns a <c>jiInferenceCancelCallback</c> function pointer representing the currently set
///   inference cancellation handler. If no handler has been set, it returns <c>nil</c>.
/// </returns>
/// <remarks>
///   Use this function to obtain the currently active inference cancellation callback. This callback
///   is invoked during an inference process to determine whether the operation should be canceled.
///   This is useful for tracking or debugging the callback configuration in the library.
/// </remarks>
function jiGetInferenceCancelCallback(): jiInferenceCancelCallback; cdecl; external JETINFERO_DLL;

/// <summary>
///   Sets a custom inference cancellation callback function.
/// </summary>
/// <param name="AHandler">
///   The callback function to be invoked during an inference process to determine if it should be canceled.
///   Pass <c>nil</c> to remove the current callback.
/// </param>
/// <param name="AUserData">
///   Custom user data passed to the callback each time it is invoked. This can be used to provide
///   additional context or application-specific state information.
/// </param>
/// <remarks>
///   This function allows you to specify a custom cancellation mechanism for inference processes.
///   The callback is invoked periodically during inference, allowing the application to cancel
///   the process under specific conditions (e.g., timeout, user intervention).
///   Ensure the callback function is properly implemented to avoid performance bottlenecks or unintended cancellations.
///   If no callback is set, the inference process will continue uninterrupted.
/// </remarks>
procedure jiSetInferenceCancelCallback(const AHandler: jiInferenceCancelCallback; const AUserData: Pointer); cdecl; external JETINFERO_DLL;

/// <summary>
///   Retrieves the current inference token callback function.
/// </summary>
/// <returns>
///   Returns a <c>jiInferenceTokenCallback</c> function pointer representing the currently set
///   token handler. If no handler has been set, it returns <c>nil</c>.
/// </returns>
/// <remarks>
///   Use this function to obtain the currently active callback for handling tokens generated
///   during an inference process. The callback is invoked each time a new token is generated,
///   allowing real-time processing or streaming of output tokens.
/// </remarks>
function jiGetInferenceTokenCallback(): jiInferenceTokenCallback; cdecl; external JETINFERO_DLL;

/// <summary>
///   Sets a custom inference token callback function.
/// </summary>
/// <param name="AHandler">
///   The callback procedure to be invoked each time a token is generated during the inference process.
///   Pass <c>nil</c> to remove the current callback.
/// </param>
/// <param name="AUserData">
///   Custom user data passed to the callback each time it is invoked. This can be used to provide
///   additional context or application-specific state information.
/// </param>
/// <remarks>
///   This function allows you to define a custom mechanism for handling tokens generated during
///   the inference process. The callback is invoked in real-time as tokens are produced, making
///   it useful for applications that require streaming responses, such as chat interfaces or live
///   transcription systems.
///
///   Ensure the callback implementation is efficient and thread-safe, as it may be invoked frequently
///   during the inference process. Removing the callback by passing <c>nil</c> will result in no tokens
///   being handled directly.
/// </remarks>
procedure jiSetInferenceTokenCallback(const AHandler: jiInferenceTokenCallback; const AUserData: Pointer); cdecl; external JETINFERO_DLL;

/// <summary>
///   Retrieves the current informational message callback function for model-related events.
/// </summary>
/// <returns>
///   Returns a <c>jiInfoCallback</c> function pointer representing the currently set
///   handler for processing informational messages about the model being loaded, its features, and metadata.
///   If no handler has been set, it returns <c>nil</c>.
/// </returns>
/// <remarks>
///   Use this function to obtain the currently active callback that processes messages related to
///   the model loading process. These messages often include details about the model's features,
///   metadata, progress updates, or errors during the load operation. Tracking this callback
///   ensures visibility into the model's initialization process.
/// </remarks>
function jiGetInfoCallback(): jiInfoCallback; cdecl; external JETINFERO_DLL;

/// <summary>
///   Sets a custom informational message callback function for model-related events.
/// </summary>
/// <param name="AHandler">
///   The callback procedure to be invoked when the JetInfero library emits informational messages
///   about the model being loaded, its features, or its metadata.
///   Pass <c>nil</c> to remove the current callback.
/// </param>
/// <param name="AUserData">
///   Custom user data passed to the callback each time it is invoked. This can be used to provide
///   additional context or associate the messages with specific operations in your application.
/// </param>
/// <remarks>
///   This function allows you to define a custom handler for processing model-related messages
///   generated by the JetInfero library. Such messages may include information about:
///   <para>- The model's name, size, and supported features.</para>
///   <para>- Metadata such as version, configuration, or training details.</para>
///   <para>- Progress updates or warnings during the model loading process.</para>
///   <para>- Errors or issues encountered during initialization.</para>
///
///   The callback enables applications to log these messages, display them to users, or trigger
///   specific actions based on the content of the messages. Ensure the implementation of the callback
///   is efficient, as it may be invoked frequently during the model's load and initialization phases.
///   Removing the callback by passing <c>nil</c> disables these notifications.
/// </remarks>
procedure jiSetInfoCallback(const AHandler: jiInfoCallback; const AUserData: Pointer); cdecl; external JETINFERO_DLL;

/// <summary>
///   Retrieves the current callback function for tracking model loading progress.
/// </summary>
/// <returns>
///   Returns a <c>jiLoadModelProgressCallback</c> function pointer representing the currently set
///   handler for tracking the progress of a model being loaded. If no handler has been set, it returns <c>nil</c>.
/// </returns>
/// <remarks>
///   Use this function to obtain the currently active callback that monitors the loading progress of a model.
///   The callback provides progress updates as the model is initialized, allowing applications to visualize
///   or log the loading status in real time.
/// </remarks>
function jiGetLoadModelProgressCallback(): jiLoadModelProgressCallback; cdecl; external JETINFERO_DLL;

/// <summary>
///   Sets a custom callback function for tracking model loading progress.
/// </summary>
/// <param name="AHandler">
///   The callback function to be invoked periodically during the model loading process. This callback
///   provides real-time updates on the progress of loading the model.
///   Pass <c>nil</c> to remove the current callback.
/// </param>
/// <param name="AUserData">
///   Custom user data passed to the callback each time it is invoked. This can provide additional
///   context or associate the progress updates with specific operations in your application.
/// </param>
/// <remarks>
///   This function allows you to define a custom handler for monitoring the progress of model loading.
///   <para>The callback is invoked with the following information:</para>
///   <para>- The name of the model being loaded (<c>AModelName</c>).</para>
///   <para>- The progress percentage (<c>AProgress</c>), represented as a floating-point value (0.0 to 100.0).</para>
///   <para>- Custom user data (<c>AUserData</c>) that can be used for additional context or state tracking.</para>
///
///   <para><b>Use Cases:</b></para>
///   <para>- Displaying a progress bar or percentage in the user interface to indicate loading status.</para>
///   <para>- Logging detailed progress updates for debugging or monitoring purposes.</para>
///   <para>- Canceling the model loading process if specific conditions are met (e.g., user interruption or timeout).</para>
///
///   Ensure the callback implementation is efficient and avoids long-running operations, as it may
///   be invoked frequently during the model loading process. Passing <c>nil</c> disables progress notifications.
/// </remarks>
procedure jiSetLoadModelProgressCallback(const AHandler: jiLoadModelProgressCallback; const AUserData: Pointer); cdecl; external JETINFERO_DLL;

/// <summary>
///   Retrieves the current callback function for handling model load completion events.
/// </summary>
/// <returns>
///   Returns a <c>jiLoadModelCallback</c> function pointer representing the currently set
///   handler for processing model load completion results. If no handler has been set, it returns <c>nil</c>.
/// </returns>
/// <remarks>
///   Use this function to obtain the currently active callback that is invoked when a model
///   has finished loading. The callback provides critical information about whether the loading
///   process was successful or encountered an error, enabling applications to respond accordingly.
/// </remarks>
function jiGetLoadModelCallback(): jiLoadModelCallback; cdecl; external JETINFERO_DLL;

/// <summary>
///   Sets a custom callback function for handling model load completion events.
/// </summary>
/// <param name="AHandler">
///   The callback procedure to be invoked when the JetInfero library finishes loading a model.
///   Pass <c>nil</c> to remove the current callback.
/// </param>
/// <param name="AUserData">
///   Custom user data passed to the callback each time it is invoked. This can provide additional
///   context or associate the model load results with specific operations in your application.
/// </param>
/// <remarks>
///   This function allows you to define a custom handler to process the results of a model
///   loading operation. The callback provides the following information:
///   <para>- The name of the model (<c>AModelName</c>) that was being loaded.</para>
///   <para>- A success flag (<c>ASuccess</c>) indicating whether the model loaded successfully (<c>True</c>)
///     or if an error occurred during the process (<c>False</c>).</para>
///   <para>- Custom user data (<c>AUserData</c>) to enable application-specific state tracking or context.</para>
///
///   <para><b>Use Cases:</b></para>
///   <para>- Logging the outcome of model loading operations for debugging or monitoring.</para>
///   <para>- Displaying success or error messages in the user interface based on the result.</para>
///   <para>- Triggering subsequent actions in your application when a model is successfully loaded, such as
///     initializing dependent systems or starting an inference process.</para>
///
///   If no callback is set (by passing <c>nil</c>), the library will not notify the application
///   of the model load results, requiring the application to manually handle success or failure checks.
///
///   <para><b>Implementation Notes:</b></para>
///   <para>- Ensure the callback function is efficient and does not block execution unnecessarily.</para>
///   <para>- If using multithreaded applications, ensure the callback is thread-safe to avoid potential issues.</para>
/// </remarks>
procedure jiSetLoadModelCallback(const AHandler: jiLoadModelCallback; const AUserData: Pointer); cdecl; external JETINFERO_DLL;

/// <summary>
///   Retrieves the current callback function that is invoked at the start of the inference loop.
/// </summary>
/// <returns>
///   Returns a <c>jiInferenceStartCallback</c> function pointer representing the currently set
///   handler for responding to the start of the inference process. If no handler has been set, it returns <c>nil</c>.
/// </returns>
/// <remarks>
///   Use this function to obtain the currently active callback that is triggered at the start
///   of the inference loop. This callback is useful for performing any setup tasks or logging
///   before the inference process begins.
/// </remarks>
function jiGetInferenceStartCallback(): jiInferenceStartCallback; cdecl; external JETINFERO_DLL;

/// <summary>
///   Sets a custom callback function to be invoked at the start of the inference loop.
/// </summary>
/// <param name="AHandler">
///   The callback procedure to be invoked at the beginning of the inference process.
///   Pass <c>nil</c> to remove the current callback.
/// </param>
/// <param name="AUserData">
///   Custom user data passed to the callback each time it is invoked. This can provide additional
///   context or associate the callback with specific operations in your application.
/// </param>
/// <remarks>
///   This function allows you to define a custom handler that is triggered at the start of
///   the inference loop. The callback is invoked before any tokens are generated and is typically
///   used to prepare for the inference process.
///
///   <para><b>Use Cases:</b></para>
///   <para>- Logging the start of the inference process for debugging or monitoring purposes.</para>
///   <para>- Initializing application-specific state or resources needed during inference.</para>
///   <para>- Displaying a loading indicator or other UI elements to notify users that inference has started.</para>
///
///   If no callback is set (by passing <c>nil</c>), the application will not be notified when the
///   inference loop begins, and any preparation tasks must be handled separately.
///
///   <para><b>Implementation Notes:</b></para>
///   <para>- Ensure the callback function is lightweight to avoid delaying the inference process.</para>
///   <para>- If using multithreaded applications, ensure the callback implementation is thread-safe.</para>
/// </remarks>
procedure jiSetInferenceStartCallback(const AHandler: jiInferenceStartCallback; const AUserData: Pointer); cdecl; external JETINFERO_DLL;

/// <summary>
///   Retrieves the current callback function that is invoked at the end of the inference process.
/// </summary>
/// <returns>
///   Returns a <c>jiInferenceEndCallback</c> function pointer representing the currently set
///   handler for responding to the end of the inference process. If no handler has been set, it returns <c>nil</c>.
/// </returns>
/// <remarks>
///   Use this function to obtain the currently active callback that is triggered after the inference loop
///   has completed. This callback is useful for performing cleanup tasks, logging, or other actions
///   required after the inference process finishes.
/// </remarks>
function jiGetInferenceEndCallback(): jiInferenceEndCallback; cdecl; external JETINFERO_DLL;

/// <summary>
///   Sets a custom callback function to be invoked at the end of the inference process.
/// </summary>
/// <param name="AHandler">
///   The callback procedure to be invoked when the inference process completes.
///   Pass <c>nil</c> to remove the current callback.
/// </param>
/// <param name="AUserData">
///   Custom user data passed to the callback each time it is invoked. This can provide additional
///   context or associate the callback with specific operations in your application.
/// </param>
/// <remarks>
///   This function allows you to define a custom handler that is triggered after the inference loop
///   has finished. The callback is invoked after all tokens have been generated, and it is typically
///   used for post-processing tasks or finalizing the results of the inference.
///
///   <para><b>Use Cases:</b><para>
///   <para>- Logging the completion of the inference process for debugging or monitoring purposes.</para>
///   <para>- Cleaning up resources or resetting application-specific state initialized at the start of the inference.</para>
///   <para>- Notifying the user or application that the inference has successfully completed.</para>
///
///   If no callback is set (by passing <c>nil</c>), the application will not be notified when the
///   inference loop ends, and post-inference tasks must be handled separately.
///
///   <para><b>Implementation Notes:</para></b>
///   <para>- Ensure the callback function is efficient and avoids long-running operations to prevent delays.</para>
///   <para>- If using multithreaded applications, ensure the callback implementation is thread-safe to avoid
///     concurrency issues.</para>
/// </remarks>
procedure jiSetInferenceEndCallback(const AHandler: jiInferenceEndCallback; const AUserData: Pointer); cdecl; external JETINFERO_DLL;

/// <summary>
///   Sets the right margin for token wrapping during the inference process.
/// </summary>
/// <param name="AMargin">
///   The right margin, specified as an integer, beyond which token lines will attempt to wrap.
/// </param>
/// <remarks>
///   This procedure configures the right margin for tokenized output during the inference process.
///   When the number of characters in a token line exceeds the specified margin, the system
///   will attempt to wrap the tokens to the next line.
///
///   <para><b>Use Cases:</b><para>
///   <para>- Controlling the appearance of output when displaying tokens in a console or text-based UI.</para>
///   <para>- Ensuring readability by preventing excessively long lines of text.</para>
///
///   <para><b>Implementation Notes:</b><para>
///   <para>- Setting the margin too low may result in frequent line breaks, which can affect the flow of output.</para>
///   <para>- This setting applies only to tokenized outputs generated by the inference process and does not
///     affect the underlying model behavior.</para>
/// </remarks>
procedure jiSetTokenRightMargin(const AMargin: Int32); cdecl; external JETINFERO_DLL;

/// <summary>
///   Sets the maximum token line length before attempting to wrap at the margin.
/// </summary>
/// <param name="ALength">
///   The maximum number of characters allowed in a single line of tokens before wrapping is applied.
/// </param>
/// <remarks>
///   This procedure specifies the maximum length of a token line during the inference process.
///   If a line exceeds the defined length, it will attempt to wrap based on the configured
///   right margin (<c>jiSetTokenRightMargin</c>).
///
///   <para><b>Use Cases:</para></b>
///   <para>- Preventing excessively long token lines that may exceed display boundaries or cause
///     readability issues in text-based interfaces.</para>
///   <para>- Controlling the format of output for specific applications or environments.</para>
///
///   <para><b>Implementation Notes:</b><para>
///   <para>- This setting works in conjunction with the right margin setting configured by <c>jiSetTokenRightMargin</c>.</para>
///   <para>- Setting a very high line length may delay wrapping, while setting it too low may result in frequent line breaks.</para>
///   <para>- Ensure that both the margin and line length settings are consistent with your application's display requirements.</para>
/// </remarks>
procedure jiSetTokenMaxLineLength(const ALength: Int32); cdecl; external JETINFERO_DLL;

/// <summary>
///   Saves the current library configuration to a file.
/// </summary>
/// <param name="AFilename">
///   The name of the file (including path) where the configuration will be saved.
/// </param>
/// <returns>
///   Returns <c>True</c> if the configuration was successfully saved; otherwise, <c>False</c>.
/// </returns>
/// <remarks>
///   This function allows you to persist the current configuration settings of the JetInfero library
///   to a file. The saved configuration can later be reloaded using the <c>jiLoadConfig</c> function,
///   enabling consistent behavior across application sessions.
///
///   <para><b>Use Cases:</b><para>
///   <para>- Backing up library settings for future use or deployment to other environments.</para>
///   <para>- Saving user-specific configurations to maintain personalized behavior between sessions.</para>
///   <para>- Exporting configuration files for debugging or sharing with other users or developers.</para>
///
///   <para><b>Implementation Notes:</para></b>
///   <para>- Ensure the provided file path is valid and writable; otherwise, the function will return <c>False</c>.</para>
///   <para>- If the configuration file already exists, it may be overwritten.</para>
///   <para>- Use <c>jiGetLastError</c> to retrieve additional details if the function fails.</para>
/// </remarks>
function jiSaveConfig(const AFilename: PWideChar): Boolean; cdecl; external JETINFERO_DLL;

/// <summary>
///   Loads library configuration from a file.
/// </summary>
/// <param name="AFilename">
///   The name of the file (including path) containing the configuration to be loaded.
/// </param>
/// <returns>
///   Returns <c>True</c> if the configuration was successfully loaded; otherwise, <c>False</c>.
/// </returns>
/// <remarks>
///   This function loads configuration settings for the JetInfero library from a file,
///   replacing the current configuration with the loaded values. It is useful for restoring
///   previously saved settings or applying preconfigured settings to the library.
///
///   <para><b>Use Cases:</b><para>
///   <para>- Restoring user-specific or default settings from a previously saved configuration file.</para>
///   <para>- Loading predefined configurations for testing or deployment in specific environments.</para>
///   <para>- Quickly switching between different configurations during runtime.</para>
///
///   <para><b>Implementation Notes:</b><para>
///   <para>- Ensure the provided file path is valid and the file is readable; otherwise, the function will return <c>False</c>.</para>
///   <para>- Invalid or incompatible configuration files may cause errors; use <c>jiGetLastError</c> to retrieve details if loading fails.<para>
///   <para>- Loading a configuration overwrites the current settings, so save any unsaved changes using <c>jiSaveConfig</c> before loading new configurations.</para>
/// </remarks>
function jiLoadConfig(const AFilename: PWideChar): Boolean; cdecl; external JETINFERO_DLL;

/// <summary>
///   Clears all previously defined models from the JetInfero library.
/// </summary>
/// <remarks>
///   This procedure removes all models defined in the JetInfero library, resetting the internal
///   model definitions. After calling this procedure, no models will be available for inference
///   until new models are defined using <c>jiDefineModel</c>.
///
///   <para><b>Use Cases:</b></para>
///   <para>- Resetting the library to a clean state when switching between configurations.</para>
///   <para>- Clearing models during cleanup before unloading the library or exiting the application.</para>
///
///   <para><b>Implementation Notes:</b></para>
///   <para>- Ensure no active inferences are using the defined models before calling this procedure.</para>
///   <para>- This action cannot be undone; all model definitions must be reloaded manually.</para>
/// </remarks>
procedure jiClearModelDefines(); cdecl; external JETINFERO_DLL;

/// <summary>
///   Defines a model for use in the JetInfero library.
/// </summary>
/// <param name="AFilename">
///   The file path of the model to be defined.
/// </param>
/// <param name="ARefName">
///   A reference name to associate with the model. This name is used to identify the model
///   within the library.
/// </param>
/// <param name="ATemplate">
///   A template string that defines the structure or formatting of the model's input prompts.
/// </param>
/// <param name="ATemplateEnd">
///   A template string that defines the structure or formatting of the model's closing or end prompts.
/// </param>
/// <param name="ACapitalizeRole">
///   A boolean value indicating whether the roles (e.g., user, assistant) should be capitalized in the model output.
///   Default is <c>False</c>.
/// </param>
/// <param name="AMaxContext">
///   The maximum context size (in tokens) that the model should maintain. Default is <c>1024</c>.
/// </param>
/// <param name="AMainGPU">
///   <para>The GPU to use for loading the model. Options are:</para>
///   <para>- <c>-1</c>: Automatically selects the best available GPU.</para>
///   <para>- <c>0-N</c>: Specifies the GPU index to use, where <c>0</c> is the first GPU and <c>N</c> is the GPU number.
///   Default is <c>-1</c>.</para>
/// </param>
/// <param name="AGPULayers">
/// Specifies how many model layers to offload to the GPU for acceleration. Options are:
///   <para>- <c>-1</c>: Offload all layers to the GPU.</para>
///   <para>- <c>0</c>: Use the CPU only (no GPU acceleration).</para>
///   <para>- <c>1-N</c>: Offload the specified number of layers to the GPU, where <c>1</c> is the first layer and <c>N</c> is the maximum layer to offload.
///   Default is <c>-1</c>.</para>
/// </param>
/// <param name="AMaxThreads">
///   The maximum number of threads to use for processing. Default is <c>4</c>.
/// </param>
/// <returns>
///   Returns <c>True</c> if the model was successfully defined; otherwise, <c>False</c>.
/// </returns>
/// <remarks>
///   This function registers a model in the JetInfero library, allowing it to be used for inference.
///   You can define multiple models with unique reference names, enabling multi-model workflows.
///
///   <para><b>GPU Settings:</b></para>
///   <para>- <c>AMainGPU</c> determines which GPU to use or whether to auto-select the best GPU.</para>
///   <para>- <c>AGPULayers</c> specifies how much of the model's computational workload is offloaded to the GPU.</para>
///   <para>  Using <c>-1</c> for all layers provides maximum acceleration if supported by the hardware.</para>
///   <para>  Use <c>0</c> for CPU-only inference, which is useful on systems without a GPU or when GPU acceleration</para>
///   <para>  is not desired.</para>
///
///   <para><b>Use Cases:</b></para>
///   <para>- Adding a model to the library for inference after clearing existing definitions.</para>
///   <para>- Configuring specific templates, context sizes, or GPU acceleration for custom workflows.</para>
///   <para>- Optimizing performance by selectively offloading computation to the GPU.</para>
///
///   <para><b>Implementation Notes:</b></para>
///   <para>- Ensure the specified file path for <c>AFilename</c> is valid and accessible.</para>
///   <para>- Reference names (<c>ARefName</c>) must be unique; attempting to define a model with a duplicate name will fail.</para>
///   <para>- Use <c>jiGetLastError</c> to retrieve error details if the function returns <c>False</c>.</para>
///   <para>- For multi-GPU systems, test various <c>AMainGPU</c> and <c>AGPULayers</c> configurations to optimize performance.</para>
/// </remarks>
function jiDefineModel(const AFilename, ARefName, ATemplate, ATemplateEnd: PWideChar; ACapitalizeRole: Boolean=False; const AMaxContext: UInt32=1024; const AMainGPU: Int32=-1; const AGPULayers: Int32=-1; const AMaxThreads: Int32=4): Boolean; cdecl; external JETINFERO_DLL;

/// <summary>
///   Removes a defined model from the JetInfero library.
/// </summary>
/// <param name="ARefName">
///   The reference name of the model to remove. This must match the name used when defining the model.
/// </param>
/// <returns>
///   Returns <c>True</c> if the model was successfully removed; otherwise, <c>False</c>.
/// </returns>
/// <remarks>
///   This function unregisters a specific model from the JetInfero library, freeing associated resources.
///   The model will no longer be available for inference after it is removed.
///
///   <para><b>Use Cases:</b></para>
///   <para>- Removing models that are no longer needed to free up memory or GPU resources.</para>
///   <para>- Dynamically switching models during runtime by removing unused models and defining new ones.</para>
///
///   <para><b>Implementation Notes:</b></para>
///   <para>- Ensure no active inferences are using the model before attempting to remove it.</para>
///   <para>- Use <c>jiGetLastError</c> to retrieve error details if the function fails.</para>
/// </remarks>
function jiRemoveModelDefine(const ARefName: PWideChar): Boolean; cdecl; external JETINFERO_DLL;

/// <summary>
///   Saves all defined models to a JSON file for external editing or backup.
/// </summary>
/// <param name="AFilename">
///   The name of the JSON file (including path) where the model definitions will be saved.
/// </param>
/// <returns>
///   Returns <c>True</c> if the model definitions were successfully saved; otherwise, <c>False</c>.
/// </returns>
/// <remarks>
///   This function serializes all currently defined models in the JetInfero library to a JSON file.
///   The saved JSON file can be edited externally or used as a backup for reloading model definitions later.
///
///   <para><b>Use Cases:</b></para>
///   <para>- Backing up model definitions to ensure reproducibility or share them with other users.</para>
///   <para>- Allowing developers or users to edit model settings externally in a structured format (JSON).</para>
///   <para>- Preparing preconfigured model definitions for deployment in other environments.</para>
///
///   <para><b>Implementation Notes:</b></para>
///   <para>- Ensure the specified file path is valid and writable; otherwise, the function will return <c>False</c>.</para>
///   <para>- The saved JSON file will include all model details, such as filenames, reference names, templates, and GPU settings.</para>
///   <para>- Use <c>jiGetLastError</c> to retrieve additional details if the function fails.</para>
/// </remarks>
function jiSaveModelDefines(const AFilename: PWideChar): Boolean; cdecl; external JETINFERO_DLL;

/// <summary>
///   Loads model definitions from a JSON file.
/// </summary>
/// <param name="AFilename">
///   The name of the JSON file (including path) containing the model definitions to load.
/// </param>
/// <returns>
///   Returns <c>True</c> if the model definitions were successfully loaded; otherwise, <c>False</c>.
/// </returns>
/// <remarks>
///   This function deserializes model definitions from a JSON file and loads them into the JetInfero library.
///   It replaces any existing model definitions with the ones specified in the file.
///
///   <para><b>Use Cases:</b></para>
///   <para>- Restoring model definitions from a previously saved JSON file.</para>
///   <para>- Dynamically reconfiguring model definitions at runtime by loading a new configuration file.</para>
///   <para>- Importing externally edited or preconfigured model definitions for use in the library.</para>
///
///   <para><b>Implementation Notes:</b></para>
///   <para>- Ensure the specified file path is valid and the file is readable; otherwise, the function will return <c>False</c>.</para>
///   <para>- The JSON file must adhere to the expected structure for model definitions, including fields such as filenames,</para>
///   <para>  reference names, templates, and GPU configurations.</para>
///   <para>- If the file contains invalid or incompatible definitions, the function may fail; use <c>jiGetLastError</c>
///   <para>  to retrieve error details.</para>
///   <para>- All existing model definitions will be cleared before loading new definitions, so ensure any necessary</para>
///   <para>  definitions are saved using <c>jiSaveModelDefines</c> prior to calling this function.</para>
/// </remarks>
function jiLoadModelDefines(const AFilename: PWideChar): Boolean; cdecl; external JETINFERO_DLL;

/// <summary>
///   Loads a model by its reference name in the JetInfero library.
/// </summary>
/// <param name="AModelRef">
///   The reference name of the model to load. This must match the reference name provided
///   when the model was defined using <c>jiDefineModel</c>.
/// </param>
/// <returns>
///   Returns <c>True</c> if the model is successfully loaded or if it is already loaded; otherwise, <c>False</c>.
/// </returns>
/// <remarks>
///   This function attempts to load a model based on its reference name. If the model is already
///   loaded, the function will return <c>True</c> without reloading it. If the model is not loaded,
///   the library will attempt to load it using the configuration defined during its registration
///   with <c>jiDefineModel</c>.
///
///   <para><b>Use Cases:</b></para>
///   <para>- Dynamically loading models for inference based on runtime requirements.</para>
///   <para>- Reusing already loaded models to avoid redundant operations and improve performance.</para>
///
///   <para><b>Implementation Notes:</b></para>
///   <para>- Ensure the reference name (<c>AModelRef</c>) matches an existing model defined using <c>jiDefineModel</c>.
///   <para>- If the model fails to load (e.g., due to missing files or incorrect configuration), the function<para>
///   <para>  will return <c>False</c>. Use <c>jiGetLastError</c> to retrieve additional details about the failure.</para>
///   <para>- Loading a model that is already loaded is a no-op and will return <c>True</c>.</para>
/// </remarks>
function jiLoadModel(const AModelRef: PWideChar): Boolean; cdecl; external JETINFERO_DLL;

/// <summary>
///   Unloads the currently loaded model in the JetInfero library.
/// </summary>
/// <remarks>
///   This procedure unloads the currently active model, freeing up memory and other resources
///   associated with it. After calling this function, no model will be available for inference
///   until another model is loaded using <c>jiLoadModel</c>.
///
///   <para><b>Use Cases:</b></para>
///   <para>- Freeing up resources when a model is no longer needed.</para>
///   <para>- Dynamically switching between models by unloading the current model before loading a new one.</para>
///
///   <para><b>Implementation Notes:</para></b>
///   <para>- Ensure no active inferences are using the model before calling this procedure to avoid undefined behavior.</para>
///   <para>- If no model is currently loaded, calling this procedure will have no effect.</para>
///   <para>- After unloading, attempting to perform inference will fail until a new model is loaded.</para>
/// </remarks>
procedure jiUnloadModel(); cdecl; external JETINFERO_DLL;

/// <summary>
///   Clears all messages in the current conversation context.
/// </summary>
/// <remarks>
///   This procedure removes all messages that have been added to the current conversation context.
///   After calling this function, no messages will be available for inference until new messages
///   are added using <c>jiAddMessage</c>.
///
///   <para><b>Use Cases:</b></para>
///   <para>- Resetting the conversation context to start a new inference session.</para>
///   <para>- Clearing old messages to free up memory or simplify the input for the LLM.</para>
///
///   <para><b>Implementation Notes:</para></b>
///   <para>- Ensure that clearing messages is appropriate for your application's workflow, as all added</para>
///   <para>  messages will be permanently removed.</para>
///   <para>- After clearing messages, the inference process will use an empty context unless new messages are added.</para>
/// </remarks>
procedure jiClearMessages(); cdecl; external JETINFERO_DLL;

/// <summary>
///   Adds a message to the current conversation context.
/// </summary>
/// <param name="ARole">
///   The role of the entity adding the message (e.g., "user", "assistant").
/// </param>
/// <param name="AContent">
///   The content of the message to be added.
/// </param>
/// <returns>
///   Returns an integer representing the total number of messages in the current conversation context after the message is added.
/// </returns>
/// <remarks>
///   This function adds a message to the conversation context, which will be used by the LLM during inference.
///   Each message consists of a role (e.g., "user", "assistant") and its associated content.
///
///   <para><b>Use Cases:</b></para>
///   <para>- Building a conversation history for context-aware inference.</para>
///   <para>- Simulating multi-turn conversations by alternating between user and assistant messages.</para>
///
///   <b>Implementation Notes:</b>
///   <para>- Ensure that <c>ARole</c> is a valid role recognized by your application (e.g., "user" or "assistant").</para>
///   <para>- If <c>AContent</c> is empty or null, the message will still be added but may not contribute meaningfully to inference.</para>
///   <para>- Adding too many messages may impact performance or exceed the maximum context size for the model.</para>
/// </remarks>
function jiAddMessage(const ARole, AContent: PWideChar): Int32; cdecl; external JETINFERO_DLL;

/// <summary>
///   Retrieves the last message added by the user.
/// </summary>
/// <returns>
///   Returns a wide character pointer (<c>PWideChar</c>) to the content of the last message added by the user.
/// </returns>
/// <remarks>
///   This function provides access to the most recent message submitted by the user in the conversation context.
///   It is useful for applications that need to confirm or display the user's latest input.
///
///   <para></b>Use Cases:</b></para>
///   <para>- Displaying the user's last message in a UI for confirmation or review.</para>
///   <para>- Debugging or logging the user's most recent input for monitoring purposes.</para>
///
///   <para><b>Implementation Notes:</b></para>
///   <para>- If no user message exists, the function may return <c>nil</c>.</para>
///   <para>- The returned message is valid until a new message is added or the context is cleared.</para>
/// </remarks>
function jiGetLastUserMessage(): PWideChar; cdecl; external JETINFERO_DLL;

/// <summary>
///   Retrieves the prompt constructed from all added messages for use in LLM inference.
/// </summary>
/// <param name="AModelRef">
///   The reference name of the model for which the prompt is being constructed. This must match the
///   reference name provided when the model was defined using <c>jiDefineModel</c>.
/// </param>
/// <returns>
///   Returns a wide character pointer (<c>PWideChar</c>) to the prompt that will be sent to the LLM for inference.
/// </returns>
/// <remarks>
///   This function constructs and retrieves the full prompt that the LLM will process during inference.
///   The prompt includes all messages currently added to the conversation context, formatted according
///   to the model's template settings.
///
///   <para><b>Use Cases:</b></para>
///   <para>- Debugging or reviewing the exact input prompt that will be sent to the LLM.</para>
///   <para>- Preprocessing or validating the constructed prompt before initiating inference.</para>
///
///   <para><b>Implementation Notes:</b></para>
///   <para>- Ensure that <c>AModelRef</c> matches an existing model defined using <c>jiDefineModel</c>.</para>
///   <para>- If no messages have been added to the context, the function may return an empty string or <c>nil</c>.</para>
///   <para>- The prompt is formatted according to the model's template settings, which can be defined during model registration.</para>
/// </remarks>
function jiGetMessagePrompt(const AModelRef: PWideChar): PWideChar; cdecl; external JETINFERO_DLL;

/// <summary>
///   Executes the inference process using the specified model and sends generated tokens
///   to the <c>InferenceTokenCallback</c> if defined.
/// </summary>
/// <param name="AModelRef">
///   The reference name of the model to use for inference. This must match the reference name
///   provided when the model was defined using <c>jiDefineModel</c>.
/// </param>
/// <returns>
///   Returns <c>True</c> if the inference process completed successfully; otherwise, <c>False</c>.
/// </returns>
/// <remarks>
///   This function runs the inference process using the conversation context defined by the
///   added messages (<c>jiAddMessage</c>). The result of the inference can be retrieved
///   using <c>jiGetInferenceResponse</c>. Additionally, if an <c>InferenceTokenCallback</c>
///   has been set using <c>jiSetInferenceTokenCallback</c>, each token generated during the
///   inference process will be sent to the callback in real-time.
///
///   <para><b>Use Cases:</b></para>
///   <para>- Generating a response from the LLM based on the conversation context.</para>
///   <para>- Streaming tokens in real-time to update a user interface or log partial responses.</para>
///   <para>- Running inference in applications that require real-time or batch processing of user inputs.</para>
///
///   <para><b>Behavior with <c>InferenceTokenCallback</c>:</b></para>
///   <para>- If a callback is defined, each token generated by the inference process will be sent</para>
///   <para>  to the callback immediately after generation.</para>
///   <para>- The callback enables real-time streaming of tokens, which is particularly useful for</para>
///   <para>  applications like chatbots or live transcription systems.</para>
///   <para>- Ensure the callback is lightweight and thread-safe to avoid performance bottlenecks.</para>
///
///   <para><b>Implementation Notes:</b></para>
///   <para>- Ensure that <c>AModelRef</c> matches an existing and loaded model.</para>
///   <para>- If the inference fails, use <c>jiGetLastError</c> to retrieve additional details about the failure.</para>
///   <para>- Inference may take varying amounts of time depending on the model, hardware, and context size.</para>
///   <para>- For applications requiring real-time feedback, define an efficient <c>InferenceTokenCallback</c></para>.
/// </remarks>
function jiRunInference(const AModelRef: PWideChar): Boolean; cdecl; external JETINFERO_DLL;

/// <summary>
///   Retrieves the response generated by the most recent inference process.
/// </summary>
/// <returns>
///   Returns a wide character pointer (<c>PWideChar</c>) containing the response generated by the LLM.
///   If no response is available, it may return <c>nil</c>.
/// </returns>
/// <remarks>
///   This function provides access to the output generated by the LLM after an inference process.
///   The response is based on the input prompt constructed from the conversation context.
///
///   <para><b>Use Cases:</b></para>
///   <para>- Displaying the LLM's response to the user in a chat interface.</para>
///   <para>- Logging or analyzing the response for debugging or monitoring purposes.</para>
///
///   <para><b>Implementation Notes:</b></para>
///   <para>- Ensure that <c>jiRunInference</c> was successfully called prior to retrieving the response.</para>
///   <para>- The returned response remains valid until a new inference is initiated or the context is cleared.</para>
///   <para>- If no response is available, the function may return <c>nil</c>, indicating an issue with the inference.</para>
/// </remarks>
function jiGetInferenceResponse(): PWideChar; cdecl; external JETINFERO_DLL;

/// <summary>
///   Retrieves performance metrics for the most recent inference process.
/// </summary>
/// <param name="ATokensPerSecond">
///   A pointer to a <c>Double</c> that will receive the number of tokens processed per second during inference.
/// </param>
/// <param name="ATotalInputTokens">
///   A pointer to an <c>Int32</c> that will receive the total number of input tokens used in the inference.
/// </param>
/// <param name="ATotalOutputTokens">
///   A pointer to an <c>Int32</c> that will receive the total number of output tokens generated by the inference.
/// </param>
/// <remarks>
///   This procedure provides detailed performance metrics for the most recent inference process,
///   allowing developers to monitor and analyze the efficiency of the LLM's operation.
///
///   <para><b>Use Cases:</b></para>
///   <para>- Monitoring and benchmarking the performance of the LLM.</para>
///   <para>- Logging performance metrics for debugging or optimization purposes.</para>
///   <para>- Identifying potential bottlenecks in the inference process based on input/output token sizes and processing speed.</para>
///
///   <para><b>Implementation Notes:</b></para>
///   <para>- Ensure that <c>jiRunInference</c> was successfully called prior to retrieving performance metrics.</para>
///   <para>- The values returned in the pointers reflect the performance of the most recent inference process only.</para>
///   <para>- If the pointers provided are null, the function will not populate the corresponding metric.</para>
/// </remarks>
procedure jiGetPerformanceResult(ATokensPerSecond: PDouble; ATotalInputTokens: PInt32; ATotalOutputTokens: PInt32); cdecl; external JETINFERO_DLL;

implementation

uses
  Math;

initialization

{$IFNDEF FPC}
  ReportMemoryLeaksOnShutdown := True;
{$ENDIF}

  SetExceptionMask(GetExceptionMask + [exOverflow, exInvalidOp]);

finalization

  jiQuit();

end.
