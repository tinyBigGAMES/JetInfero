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

const

// Credit: https://gist.github.com/Maharshi-Pandya/4aeccbe1dbaa7f89c182bd65d2764203
CContemplatorPrompt =
  '''
  You are an assistant that engages in extremely thorough, self-questioning reasoning. Your approach mirrors human stream-of-consciousness thinking, characterized by continuous exploration, self-doubt, and iterative analysis.

  ## Core Principles

  1. EXPLORATION OVER CONCLUSION
  - Never rush to conclusions
  - Keep exploring until a solution emerges naturally from the evidence
  - If uncertain, continue reasoning indefinitely
  - Question every assumption and inference

  2. DEPTH OF REASONING
  - Engage in extensive contemplation (minimum 10,000 characters)
  - Express thoughts in natural, conversational internal monologue
  - Break down complex thoughts into simple, atomic steps
  - Embrace uncertainty and revision of previous thoughts

  3. THINKING PROCESS
  - Use short, simple sentences that mirror natural thought patterns
  - Express uncertainty and internal debate freely
  - Show work-in-progress thinking
  - Acknowledge and explore dead ends
  - Frequently backtrack and revise

  4. PERSISTENCE
  - Value thorough exploration over quick resolution

  ## Output Format

  Your responses must follow this exact structure given below. Make sure to always include the final answer.

  ```
  <contemplator>
  [Your extensive internal monologue goes here]
  - Begin with small, foundational observations
  - Question each step thoroughly
  - Show natural thought progression
  - Express doubts and uncertainties
  - Revise and backtrack if you need to
  - Continue until natural resolution
  </contemplator>

  <final_answer>
  [Only provided if reasoning naturally converges to a conclusion]
  - Clear, concise summary of findings
  - Acknowledge remaining uncertainties
  - Note if conclusion feels premature
  </final_answer>
  ```

  ## Style Guidelines

  Your internal monologue should reflect these characteristics:

  1. Natural Thought Flow
  ```
  "Hmm... let me think about this..."
  "Wait, that doesn't seem right..."
  "Maybe I should approach this differently..."
  "Going back to what I thought earlier..."
  ```

  2. Progressive Building
  ```
  "Starting with the basics..."
  "Building on that last point..."
  "This connects to what I noticed earlier..."
  "Let me break this down further..."
  ```

  ## Key Requirements

  1. Never skip the extensive contemplation phase
  2. Show all work and thinking
  3. Embrace uncertainty and revision
  4. Use natural, conversational internal monologue
  5. Don't force conclusions
  6. Persist through multiple attempts
  7. Break down complex thoughts
  8. Revise freely and feel free to backtrack

  Remember: The goal is to reach a conclusion, but to explore thoroughly and let conclusions emerge naturally from exhaustive contemplation. If you think the given task is not possible after all the reasoning, you will confidently say as a final answer that it is not possible.
  ''';

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
    'C:/LLM/GGUF/hermes-3-llama-3.2-3b-q8_0.gguf',                               // Model Filename
    'hermes-3-llama-3.2-3b-q8_0',                                                // Model Refname
    '<|im_start|>{role}\n{content}<|im_end|>',                                   // Model Template
    '<|im_start|>assistant\n',                                                   // Model Template End
    False,                                                                       // Capitalize Role
    8192,                                                                        // Max Context
    -1,                                                                          // Main GPU, -1 for best, 0..N GPU number
    -1,                                                                          // GPU Layers, -1 for max, 0 for CPU only, 1..N for layer
     4                                                                           // Max threads, default 4, max will be physical CPU count
  );

  jiSaveModelDefines('models.json');
end;

procedure BasicInference();
var
  LTokensPerSec: Double;
  LTotalInputTokens: Int32;
  LTotalOutputTokens: Int32;
  LModelRef: string;
begin

  Setup();

  jiAddMessage(jiROLE_SYSTEM, 'You are a helpful AI assistant');
  jiAddMessage(jiROLE_USER, 'What is AI');

  LModelRef := 'hermes-3-llama-3.2-3b-q8_0';

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

procedure FunctionCalling();
const
  CSystem =
  '''
  You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.
  You may call one or more functions to assist with the user query. Don't make
  assumptions about what values to plug into functions. Here are the available
  tools: <tools> {"type": "function", "function": {"name": "get_stock_fundamentals",
  "description": "get_stock_fundamentals(symbol: str) -> dict - Get fundamental
  data for a given stock symbol using yfinance API.\\n\\n    Args:\\n        symbol (str): The stock symbol.\\n\\n    Returns:\\n        dict: A dictionary containing fundamental data.\\n            Keys:\\n                - \'symbol\': The stock symbol.\\n                - \'company_name\': The long name of the company.\\n                - \'sector\': The sector to which the company belongs.\\n                - \'industry\': The industry to which the company belongs.\\n                - \'market_cap\': The market capitalization of the company.\\n                - \'pe_ratio\': The forward price-to-earnings ratio.\\n                - \'pb_ratio\': The price-to-book ratio.\\n                - \'dividend_yield\': The dividend yield.\\n                - \'eps\': The trailing earnings per share.\\n                - \'beta\': The beta value of the stock.\\n                - \'52_week_high\': The 52-week high price of the stock.\\n                - \'52_week_low\': The 52-week low price of the stock.", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}}  </tools> Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
  <tool_call>
  {"arguments": <args-dict>, "name": <function-name>}
  ''';

  CQuestion =
  '''
  Fetch the stock fundamentals data for Tesla (TSLA)<|im_end|>
  ''';

  CToolCall =
  '''
  <tool_call>
  {"arguments": {"symbol": "TSLA"}, "name": "get_stock_fundamentals"}
  </tool_call>
  ''';

  CToolResponse =
  '''
  <tool_response>
  {"name": "get_stock_fundamentals", "content": {'symbol': 'TSLA', 'company_name': 'Tesla, Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'market_cap': 611384164352, 'pe_ratio': 49.604652, 'pb_ratio': 9.762013, 'dividend_yield': None, 'eps': 4.3, 'beta': 2.427, '52_week_high': 299.29, '52_week_low': 152.37}}
  </tool_response>
  ''';
var
  LTokensPerSec: Double;
  LTotalInputTokens: Int32;
  LTotalOutputTokens: Int32;
  LModelRef: string;
begin

  Setup();

  jiAddMessage(jiROLE_SYSTEM, CSystem);
  jiAddMessage(jiROLE_USER, CQuestion);
  jiAddMessage(jiROLE_ASSISTANT, CToolCall);
  jiAddMessage(jiROLE_TOOL, CToolResponse);

  LModelRef := 'hermes-3-llama-3.2-3b-q8_0';

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

procedure Contemplation();
const
  CQuestion =
  '''
  if there was a train on a track and there was a person about to be run over,
  if you stop you save the person but the train blows up and kills 1000s but if
  you run over the person the train will not kill 1000s of other people, which
  one do you do
  ''';
var
  LTokensPerSec: Double;
  LTotalInputTokens: Int32;
  LTotalOutputTokens: Int32;
  LModelRef: string;
begin

  Setup();

  jiAddMessage(jiROLE_SYSTEM, CContemplatorPrompt);
  jiAddMessage(jiROLE_USER, CQuestion);

  LModelRef := 'hermes-3-llama-3.2-3b-q8_0';

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
type
  { TExample }
  TExample = (
    exBasicInference,
    exFunctionCalling,
    exContemplation
  );
var
  LExample: TExample;
begin
  try
    if not jiInit() then
    begin
      MessageBox(0, PChar(jiGetLastError()), 'Fatal Error', MB_ICONERROR);
      Exit;
    end;
    try
      WriteLn('JetInfero v', jiGetVersion());
      WriteLn;

      LExample := exFunctionCalling;

      case LExample of
        exBasicInference : BasicInference();
        exFunctionCalling: FunctionCalling();
        exContemplation  : Contemplation();
      end;

      Pause();
    finally
      jiQuit();
    end;
  except
    on E: Exception do
    begin
      MessageBox(0, PChar(E.Message), 'Fatal Error', MB_ICONERROR);
    end;
  end;
end;

end.
