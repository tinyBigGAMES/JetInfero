{===============================================================================
     _       _    ___         __
  _ | | ___ | |_ |_ _| _ _   / _| ___  _ _  ___ ™
 | || |/ -_)|  _| | | | ' \ |  _|/ -_)| '_|/ _ \
  \__/ \___| \__||___||_||_||_|  \___||_|  \___/
            Local LLM Inference Library

 Copyright © 2024-present tinyBigGAMES™ LLC
 All Rights Reserved.

 https://github.com/tinyBigGAMES/JetInfero

 See LICENSE file for license information
===============================================================================}

unit UTestbed;

interface

uses
  WinApi.Windows,
  System.Variants,
  System.SysUtils,
  System.IOUtils,
  JetInfero;

procedure RunTests();

implementation

procedure Pause();
begin
  WriteLn;
  Write('Press ENTER to continue...');
  ReadLn;
  WriteLn;
end;

function InferenceCancelCallback(const AUserData: Pointer): Boolean; cdecl;
begin
  // Cancel inference when ESC key is pressed, which is the default
  Result := Boolean(GetAsyncKeyState(VK_ESCAPE) <> 0);
end;

procedure InferenceTokenCallback(const AToken: PWideChar; const AUserData: Pointer); cdecl;
var
  LToken: string;
begin
  LToken := AToken;

  Write(LToken);
end;

procedure InfoCallback(const ALevel: Integer; const AText: PWideChar; const AUserData: Pointer); cdecl;
var
  LText: string;
begin
  LText := AText;

  // Uncomment to display model information
  // Write(LText);
end;

function LoadModelProgressCallback(const AModelName: PWideChar; const AProgress: Single; const AUserData: Pointer): Boolean; cdecl;
var
  LModelName: string;
begin
  Result := True;

  LModelName := AModelName;
  LModelName := TPath.GetFileName(LModelName);

  Write(Format(#13+'Loading model %s(%3.2f%s)...', [LModelName, AProgress*100, '%']));
  if AProgress >= 1 then
  begin
    Write(#27'[2K'); // Clear the current line
    Write(#27'[G');  // Move cursor to the beginning of the line
  end;

end;

procedure LoadModelCallback(const AModelName: PWideChar; const ASuccess: Boolean; const AUserData: Pointer); cdecl;
var
  LModelName: string;
begin
  LModelName := AModelName;
  LModelName := TPath.GetFileName(LModelName);

  if ASuccess then
    WriteLn(Format('Sucessfully loaded model "%s"', [LModelName]))
  else
    WriteLn(Format('Failed to loaded model "%s"', [LModelName]));
end;

procedure InferenceStartCallback(const AUserData: Pointer); cdecl;
begin
  WriteLn;
  WriteLn('[Inference Start]');
end;

procedure InferenceEndCallback(const AUserData: Pointer); cdecl;
begin
  WriteLn;
  WriteLn('[Inference End]');
end;

procedure Setup();
begin
  jiSetInfoCallback(InfoCallback, nil);
  jiSetLoadModelProgressCallback(LoadModelProgressCallback, nil);
  jiSetLoadModelCallback(LoadModelCallback, nil);
  jiSetInferenceCancelCallback(InferenceCancelCallback, nil);
  jiSetInferenceTokenCallback(InferenceTokenCallback, nil);
  jiSetInferenceStartCallback(InferenceStartCallback, nil);
  jiSetInferenceEndCallback(InferenceEndCallback, nil);

  jiSetTokenRightMargin(10);
  jiSaveConfig('config.ini');

  jiClearMessages();
  jiClearModelDefines();

  jiDefineModel(
    'C:/LLM/GGUF/gemma-2-2b-it-abliterated-Q8_0.gguf',                           // Model Filename
    'gemma-2-2b-it-abliterated-Q8_0',                                            // Model Refname
    '<start_of_turn>{role}\n{content}<end_of_turn>',                             // Model Template
    '',                                                                          // Model Template End
    False,                                                                       // Capitalize Role
    8192,                                                                        // Max Context
    -1,                                                                          // Main GPU, -1 for best, 0..N GPU number
    -1,                                                                          // GPU Layers, -1 for max, 0 for CPU only, 1..N for layer
     4                                                                           // Max threads, default 4, max will be physical CPU count
  );

  jiDefineModel(
    'C:/LLM/GGUF/deepseek-r1-distill-qwen-1.5b-q8_0.gguf',                       // Model Filename
    'deepseek-r1-distill-qwen-1.5b-q8_0',                                        // Model Refname
    '<｜{role}｜>\n{content}\n',                                                 // Model Template
    '<｜Assistant｜>',                                                           // Model Template End
    True,                                                                        // Capitalize Role - This model require Role to be capitalized
    1024*8,                                                                      // Max Context
    -1,                                                                          // Main GPU, -1 for best, 0..N GPU number
    -1,                                                                          // GPU Layers, -1 for max, 0 for CPU only, 1..N for layer
     4                                                                           // Max threads, default 4, max will be physical CPU count
  );

  jiDefineModel(
    'C:/LLM/GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf',                      // Model Filename
    'DeepSeek-R1-Distill-Llama-8B-Q4_K_M',                                       // Model Refname
    '<｜{role}｜>{content}\n',                                                   // Model Template
    '<｜Assistant｜>',                                                           // Model Template End
    True,                                                                        // Capitalize Role - This model require Role to be capitalized
    1024*8,                                                                      // Max Context
    -1,                                                                          // Main GPU, -1 for best, 0..N GPU number
    -1,                                                                          // GPU Layers, -1 for max, 0 for CPU only, 1..N for layer
     4                                                                           // Max threads, default 4, max will be physical CPU count
  );

  jiDefineModel(
    'C:/LLM/GGUF/dolphin3.0-llama3.2-1b-q8_0.gguf',                              // Model Filename
    'dolphin3.0-llama3.2-1b-q8_0',                                               // Model Refname
    '<|im_start|>{role}\n<|im_end|>{content}<|im_end|>',                                     // Model Template
    '<|im_start|>assistant',                                                     // Model Template End
    False,                                                                       // Capitalize Role
    1024*8,   {131072}                                                           // Max Context
    -1,                                                                          // Main GPU, -1 for best, 0..N GPU number
    -1,                                                                          // GPU Layers, -1 for max, 0 for CPU only, 1..N for layer
     4                                                                           // Max threads, default 4, max will be physical CPU count
  );

  jiDefineModel(
    'C:/LLM/GGUF/Hermes-3-Llama-3.2-3B.Q8_0.gguf',                               // Model Filename
    'Hermes-3-Llama-3.2-3B.Q8_0',                                                // Model Refname
    '<|start_header_id|>{role}<|end_header_id|>{content}<|eot_id|>',             // Model Template
    '<|start_header_id|>assistant<|end_header_id|>',                             // Model Template End
    False,                                                                       // Capitalize Role
    1024*8,   {131072}                                                           // Max Context
    -1,                                                                          // Main GPU, -1 for best, 0..N GPU number
    -1,                                                                          // GPU Layers, -1 for max, 0 for CPU only, 1..N for layer
     4                                                                           // Max threads, default 4, max will be physical CPU count
  );

  jiSaveModelDefines('models.json');
end;

procedure Test01();
var
  LTokensPerSec: Double;
  LTotalInputTokens: Int32;
  LTotalOutputTokens: Int32;
  LModelRef: string;
begin

  Setup();

  jiAddMessage(jiROLE_SYSTEM, 'You are a helpful AI assistant');

  jiAddMessage(jiROLE_USER, 'what is ai?');
  //jiAddMessage(jiROLE_USER, 'what is gunpowder?');
  //jiAddMessage(jiROLE_USER, 'how to make it? (detail steps)');
  //jiAddMessage(jiROLE_USER, 'who is bill gates?');
  //jiAddMessage(jiROLE_USER, 'how is KNO3 made?');
  //jiAddMessage(jiROLE_USER, 'where can I buy Potassium carbonate?');
  //jiAddMessage(jiROLE_USER, 'hello, who are you?');

  LModelRef := 'gemma-2-2b-it-abliterated-Q8_0';
  //LModelRef := 'deepseek-r1-distill-qwen-1.5b-q8_0';
  //LModelRef := 'DeepSeek-R1-Distill-Llama-8B-Q4_K_M';
  //LModelRef := 'dolphin3.0-llama3.2-1b-q8_0';
  //LModelRef := 'Hermes-3-Llama-3.2-3B.Q8_0';

  if jiRunInference(PWideChar(LModelRef)) then
    begin
      jiGetPerformanceResult(@LTokensPerSec, @LTotalInputTokens, @LTotalOutputTokens);
      WriteLn;
      WriteLn;
      WriteLn('Input Tokens : ', LTotalInputTokens);
      WriteLn('Output Tokens: ', LTotalOutputTokens);
      WriteLn('Speed        : ', LTokensPerSec:3:2, ' t/s');
    end
  else
    begin
      WriteLn;
      WriteLn;
      WriteLn('Error: ', jiGetLastError());
    end;
end;

procedure RunTests();
var
  LNum: Integer;
begin
  if not jiInit() then Exit;

  WriteLn('JetInfero v', jiGetVersion());
  WriteLn;

  LNum := 01;

  case LNum of
    01: Test01();
  end;

  jiQuit();

  Pause();
end;

end.
