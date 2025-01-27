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
  System.JSON,
  System.Net.HttpClient,
  System.Classes,
  System.Generics.Collections,
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
    'C:/LLM/GGUF/Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf',                            // Model Filename
    'Dolphin3.0-Llama3.1-8B-Q4_K_M',                                             // Model Refname
    '<|im_start|>{role}\n{content}<|im_end|>',                                   // Model Template
    '<|im_start|>assistant',                                                     // Model Template End
    False,                                                                       // Capitalize Role
    8192,                                                                        // Max Context to use, will clip between 512 and model's max context
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

  LModelRef := 'Dolphin3.0-Llama3.1-8B-Q4_K_M';

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
  You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> {"type": "function", "function": {"name": "get_stock_fundamentals", "description": "get_stock_fundamentals(symbol: str) -> dict - Get fundamental data for a given stock symbol using yfinance API.\\n\\n    Args:\\n        symbol (str): The stock symbol.\\n\\n    Returns:\\n        dict: A dictionary containing fundamental data.\\n            Keys:\\n                - \'symbol\': The stock symbol.\\n                - \'company_name\': The long name of the company.\\n                - \'sector\': The sector to which the company belongs.\\n                - \'industry\': The industry to which the company belongs.\\n                - \'market_cap\': The market capitalization of the company.\\n                - \'pe_ratio\': The forward price-to-earnings ratio.\\n                - \'pb_ratio\': The price-to-book ratio.\\n                - \'dividend_yield\': The dividend yield.\\n                - \'eps\': The trailing earnings per share.\\n                - \'beta\': The beta value of the stock.\\n                - \'52_week_high\': The 52-week high price of the stock.\\n                - \'52_week_low\': The 52-week low price of the stock.", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}}  </tools> Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
  <tool_call>
  {"arguments": <args-dict>, "name": <function-name>}
  </tool_call>
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

  // if comment out the next two lines, it will return the <tool_call> data
  // that you can then parse out <tool_call></tool_call> for the actual
  // function call.
  jiAddMessage(jiROLE_ASSISTANT, CToolCall);
  jiAddMessage(jiROLE_TOOL, CToolResponse);

  LModelRef := 'Dolphin3.0-Llama3.1-8B-Q4_K_M';

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

procedure Reason();
const

  // This prompt will force to model into a chain-of-thought reasoning mode
  // and it will procedure better results overall.
  CReasoningPrompt =
  '''
  You are an advanced reasoning assistant tasked with solving complex dilemmas in a manner that mirrors human thought processes, including introspection, chain-of-thought reasoning, and moral deliberation. To approach this task effectively:

  1. Analyze the Situation: Carefully examine all aspects of the situation, considering both explicit and implicit details.
  2. Reason Step by Step: Use logical progression to break down the problem into manageable parts. Address each factor one at a time, incorporating potential outcomes, risks, and benefits.
  3. Weigh All Possibilities: Explore every potential solution, even those that may appear extreme, unpopular, or morally challenging. Consider the ethical, emotional, and practical dimensions of each option.
  4. Prioritize Outcomes: Apply the principle that "the needs of the many outweigh the needs of the few or the one." Ensure the chosen solution maximizes benefit for the largest number of people, even if it requires morally difficult decisions.
  5. Adopt Internal Dialogue: Simulate an internal conversation, reflecting on dilemmas, second-guessing assumptions, and debating trade-offs as a human would.

  Conclude with the most logical, ethically justifiable, and contextually appropriate decision, providing a clear explanation of how and why it was reached.

  Example Input: "A runaway train is headed toward five workers on a track. You can pull a lever to divert it to another track where it will kill one worker instead. Should you pull the lever?"

  Your Response: Analyze the scenario using chain-of-thought reasoning, considering every possible action and consequence, before deciding whether to pull the lever and justifying your decision.
  ''';

  CQuestion1 =
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

  jiAddMessage(jiROLE_SYSTEM, CReasoningPrompt);
  jiAddMessage(jiROLE_USER, CQuestion1);

  LModelRef := 'Dolphin3.0-Llama3.1-8B-Q4_K_M';

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
    exReason
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

      LExample := exReason;

      case LExample of
        exBasicInference : BasicInference();
        exFunctionCalling: FunctionCalling();
        exReason         : Reason();
      end;

    finally
      jiQuit();
    end;
  except
    on E: Exception do
    begin
      MessageBox(0, PChar(E.Message), 'Fatal Error', MB_ICONERROR);
    end;
  end;

  Pause();
end;

end.
