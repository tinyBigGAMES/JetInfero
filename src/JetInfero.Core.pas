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

unit JetInfero.Core;

{$I JetInfero.Defines.inc}

interface

uses
  WinApi.Windows,
  System.Generics.Collections,
  System.SysUtils,
  System.StrUtils,
  System.IOUtils,
  System.Classes,
  System.Math,
  System.JSON,
  System.IniFiles,
  JetInfero.CLibs,
  JetInfero.Utils;

type

  jiInferenceCancelCallback = function(const AUserData: Pointer): Boolean; cdecl;
  jiInferenceTokenCallback = procedure(const AToken: PWideChar; const AUserData: Pointer); cdecl;
  jiInfoCallback = procedure(const ALevel: Integer; const AText: PWideChar; const AUserData: Pointer); cdecl;
  jiLoadModelProgressCallback = function(const AModelName: PWideChar; const AProgress: Single; const AUserData: Pointer): Boolean; cdecl;
  jiLoadModelCallback = procedure(const AModelName: PWideChar; const ASuccess: Boolean; const AUserData: Pointer); cdecl;
  jiInferenceStartCallback = procedure(const AUserData: Pointer); cdecl;
  jiInferenceEndCallback = procedure(const AUserData: Pointer); cdecl;

  { TModel }
  TModel = record
    Filename: string;
    RefName: string;
    MaxContext: UInt32;
    MaxThreads: Int32;
    MainGPU: Int32;
    GPULayers: Int32;
    Template: string;
    TEmplateEnd: string;
    CapitalizeRole: Boolean;
    function ToJSON(): TJSONObject;
    procedure FromJSON(const AJSON: TJSONObject);
  end;

  { TJetInfero }
  TJetInfero = class
  public type
    PerformanceResult = record
      TokensPerSecond: Double;
      TotalInputTokens: Int32;
      TotalOutputTokens: Int32;
    end;
  private type
    TMessage = record
      Role: string;
      Content: string;
    end;

    TMessages = TList<TMessage>;
    TModels = TDictionary<string, TModel>;

    TInference = record
      Active: Boolean;
      Prompt: string;
      Response: string;
      Perf: TJetInfero.PerformanceResult;
      Model: TModel;
    end;

    TCallbacks = record
      InferenceCancel: TCallback<jiInferenceCancelCallback>;
      InferenceToken: TCallback<jiInferenceTokenCallback>;
      Info: TCallback<jiInfoCallback>;
      LoadModelProgress: TCallback<jiLoadModelProgressCallback>;
      LoadModel: TCallback<jiLoadModelCallback>;
      InferenceStart: TCallback<jiInferenceStartCallback>;
      InferenceEnd: TCallback<jiInferenceEndCallback>;
    end;
  private
    FCLibsHandle: THandle;
    FClibsDLLFilename: string;
    FLastError: string;
    FVersion: string;
    FMessages: TMessages;
    FModels: TModels;
    FModel: Pllama_model;
    FInference: TInference;
    FLastUserMessage: string;
    FTokenRightMargin: Integer;
    FTokenMaxLineLength: Integer;
    FTokenResponse: TTokenResponse;
    FCallbacks: TCallbacks;
    function GetClibsDLLVersion(): string;
    function LoadDLL(): Boolean;
    procedure UnloadDLL();
    function  TokenToPiece(const AVocab: Pllama_vocab; const AContext: Pllama_context; const AToken: llama_token; const ASpecial: Boolean): string;
    function  CalcPerformance(const AContext: Pllama_context): TJetInfero.PerformanceResult;
    procedure OnNextToken(const AToken: string);
    function  OnInferenceCancel(): Boolean;
    procedure OnInfo(const ALevel: Integer; const AText: string);
    function  OnLoadModelProgress(const AModelRef: string; const AProgress: Single): Boolean;
    procedure OnLoadModel(const AModelRef: string; const ASuccess: Boolean);
    procedure OnInferenceStart();
    procedure OnInferenceEnd();
  public
    constructor Create(); virtual;
    destructor Destroy(); override;
    function  Startup(): Boolean;
    procedure Shutdown();
    procedure SetError(const AText: string; const AArgs: array of const);
    function  GetLastError(): string;
    function  GetVersion(): string;

    function  GetInferenceCancelCallback(): jiInferenceCancelCallback;
    procedure SetInferenceCancelCallback(const AHandler: jiInferenceCancelCallback; const AUserData: Pointer);

    function  GetInferenceTokenCallback(): jiInferenceTokenCallback;
    procedure SetInferenceTokenlCallback(const AHandler: jiInferenceTokenCallback; const AUserData: Pointer);

    function  GetInfoCallback(): jiInfoCallback;
    procedure SetInfoCallback(const AHandler: jiInfoCallback; const AUserData: Pointer);

    function  GetLoadModelProgressCallback(): jiLoadModelProgressCallback;
    procedure SetLoadModelProgressCallback(const AHandler: jiLoadModelProgressCallback; const AUserData: Pointer);

    function  GetLoadModelCallback(): jiLoadModelCallback;
    procedure SetLoadModelCallback(const AHandler: jiLoadModelCallback; const AUserData: Pointer);

    function  GetInferenceStartCallback(): jiInferenceStartCallback;
    procedure SetInferenceStartCallback(const AHandler: jiInferenceStartCallback; const AUserData: Pointer);

    function  GetInferenceEndCallback(): jiInferenceEndCallback;
    procedure SetInferenceEndCallback(const AHandler: jiInferenceEndCallback; const AUserData: Pointer);

    procedure SetTokenRightMargin(const AMargin: Int32);
    procedure SetTokenMaxLineLength(const ALength: Int32);

    function  SaveConfig(const AFilename: string): Boolean;
    function  LoadConfig(const AFilename: string): Boolean;

    procedure ClearModelDefines();
    function  DefineModel(const AFilename, ARefName, ATemplate, ATemplateEnd: string; ACapitalizeRole: Boolean=False; const AMaxContext: UInt32=1024; const AMainGPU: Int32=-1; const AGPULayers: Int32=-1; const AMaxThreads: Int32=4): Boolean;
    function  RemoveModelDefine(const ARefName: string): Boolean;
    function  SaveModelDefines(const AFilename: string): Boolean;
    function  LoadModelDefines(const AFilename: string): Boolean;

    function  LoadModel(const AModelRef: string): Boolean;
    procedure UnloadModel();

    procedure ClearMessages();
    function  AddMessage(const ARole, AContent: string): Int32;
    function  GetLastUserMessage(): string;
    function  GetMessagePrompt(const AModelRef: string): string;

    function  RunInference(const AModelRef: string): Boolean;
    function  GetInferenceResponse(): string;
    function  GetPerformanceResult(): TJetInfero.PerformanceResult;
  end;

//=== EXPORTS ===============================================================
function  jiInit(): Boolean; cdecl; exports jiInit;
function  jiWasInit(): Boolean; cdecl; exports jiWasInit;
function  jiGetVersion(): PWideChar; cdecl; exports jiGetVersion;
procedure jiQuit(); cdecl; exports jiQuit;
function  jiGetLastError(): PWideChar; cdecl; exports jiGetLastError;
function  jiGetInferenceCancelCallback(): jiInferenceCancelCallback; cdecl; exports jiGetInferenceCancelCallback;
procedure jiSetInferenceCancelCallback(const AHandler: jiInferenceCancelCallback; const AUserData: Pointer); cdecl; exports jiSetInferenceCancelCallback;
function  jiGetInferenceTokenCallback(): jiInferenceTokenCallback; cdecl; exports jiGetInferenceTokenCallback;
procedure jiSetInferenceTokenCallback(const AHandler: jiInferenceTokenCallback; const AUserData: Pointer); cdecl; exports jiSetInferenceTokenCallback;
function  jiGetInfoCallback(): jiInfoCallback; cdecl; exports jiGetInfoCallback;
procedure jiSetInfoCallback(const AHandler: jiInfoCallback; const AUserData: Pointer); cdecl; exports jiSetInfoCallback;
function  jiGetLoadModelProgressCallback(): jiLoadModelProgressCallback; cdecl; exports jiGetLoadModelProgressCallback;
procedure jiSetLoadModelProgressCallback(const AHandler: jiLoadModelProgressCallback; const AUserData: Pointer); cdecl; exports jiSetLoadModelProgressCallback;
function  jiGetLoadModelCallback(): jiLoadModelCallback; cdecl; exports jiGetLoadModelCallback;
procedure jiSetLoadModelCallback(const AHandler: jiLoadModelCallback; const AUserData: Pointer); cdecl; exports jiSetLoadModelCallback;
function  jiGetInferenceStartCallback(): jiInferenceStartCallback; cdecl; exports jiGetInferenceStartCallback;
procedure jiSetInferenceStartCallback(const AHandler: jiInferenceStartCallback; const AUserData: Pointer); cdecl; exports jiSetInferenceStartCallback;
function  jiGetInferenceEndCallback(): jiInferenceEndCallback; cdecl; exports jiGetInferenceEndCallback;
procedure jiSetInferenceEndCallback(const AHandler: jiInferenceEndCallback; const AUserData: Pointer); cdecl; exports jiSetInferenceEndCallback;
procedure jiSetTokenRightMargin(const AMargin: Int32); cdecl; exports jiSetTokenRightMargin;
procedure jiSetTokenMaxLineLength(const ALength: Int32); cdecl; exports jiSetTokenMaxLineLength;
function  jiSaveConfig(const AFilename: PWideChar): Boolean; cdecl; exports jiSaveConfig;
function  jiLoadConfig(const AFilename: PWideChar): Boolean; cdecl; exports jiLoadConfig;
procedure jiClearModelDefines(); cdecl; exports jiClearModelDefines;
function  jiDefineModel(const AFilename, ARefName, ATemplate, ATemplateEnd: PWideChar; ACapitalizeRole: Boolean=False; const AMaxContext: UInt32=1024; const AMainGPU: Int32=-1; const AGPULayers: Int32=-1; const AMaxThreads: Int32=4): Boolean; cdecl; exports jiDefineModel;
function  jiRemoveModelDefine(const ARefName: PWideChar): Boolean; cdecl; exports jiRemoveModelDefine;
function  jiSaveModelDefines(const AFilename: PWideChar): Boolean; cdecl; exports jiSaveModelDefines;
function  jiLoadModelDefines(const AFilename: PWideChar): Boolean; cdecl; exports jiLoadModelDefines;
function  jiLoadModel(const AModelRef: PWideChar): Boolean; cdecl; exports jiLoadModel;
procedure jiUnloadModel(); cdecl; exports jiUnloadModel;
procedure jiClearMessages(); cdecl; exports jiClearMessages;
function  jiAddMessage(const ARole, AContent: PWideChar): Int32; cdecl; exports jiAddMessage;
function  jiGetLastUserMessage(): PWideChar; cdecl; exports jiGetLastUserMessage;
function  jiGetMessagePrompt(const AModelRef: PWideChar): PWideChar; cdecl; exports jiGetMessagePrompt;
function  jiRunInference(const AModelRef: PWideChar): Boolean; cdecl; exports jiRunInference;
function  jiGetInferenceResponse(): PWideChar; cdecl; exports jiGetInferenceResponse;
procedure jiGetPerformanceResult(ATokensPerSecond: PDouble; ATotalInputTokens: PInt32; ATotalOutputTokens: PInt32); cdecl; exports jiGetPerformanceResult;

implementation

var
  LJetInfero: TJetInfero = nil;

//=== EXPORTS ===============================================================
function  jiInit(): Boolean;
begin
  Result := False;
  if not Assigned(LJetInfero) then
  begin
    LJetInfero := TJetInfero.Create();
    Result := LJetInfero.Startup();
  end;
end;

function  jiWasInit(): Boolean;
begin
  Result := False;
  if Assigned(LJetInfero) then
  begin
    Result := True;
  end;
end;

function jiGetVersion(): PWideChar;
begin
  Result := nil;
  if not Assigned(LJetInfero) then Exit;

  Result := PWideChar(LJetInfero.GetVersion());
end;

procedure jiQuit();
begin
  if Assigned(LJetInfero) then
  begin
    LJetInfero.Free();
    LJetInfero := nil;
  end;
end;

function  jiGetLastError(): PWideChar;
begin
  Result := nil;
  if Assigned(LJetInfero) then
  begin
    Result := PWideChar(LJetInfero.GetLastError());
  end;
end;

function  jiGetInferenceCancelCallback(): jiInferenceCancelCallback;
begin
  Result := nil;

  if Assigned(LJetInfero) then
  begin
    Result := LJetInfero.GetInferenceCancelCallback();
  end;
end;

procedure jiSetInferenceCancelCallback(const AHandler: jiInferenceCancelCallback; const AUserData: Pointer);
begin
  if Assigned(LJetInfero) then
  begin
    LJetInfero.SetInferenceCancelCallback(AHandler, AUserData);
  end;
end;

function  jiGetInferenceTokenCallback(): jiInferenceTokenCallback;
begin
  Result := nil;

  if Assigned(LJetInfero) then
  begin
    Result := LJetInfero.GetInferenceTokenCallback();
  end;
end;

procedure jiSetInferenceTokenCallback(const AHandler: jiInferenceTokenCallback; const AUserData: Pointer);
begin
  if Assigned(LJetInfero) then
  begin
    LJetInfero.SetInferenceTokenlCallback(AHandler, AUserData);
  end;
end;

function  jiGetInfoCallback(): jiInfoCallback;
begin
  Result := nil;

  if Assigned(LJetInfero) then
  begin
    Result := LJetInfero.GetInfoCallback();
  end;
end;

procedure jiSetInfoCallback(const AHandler: jiInfoCallback; const AUserData: Pointer);
begin
  if Assigned(LJetInfero) then
  begin
    LJetInfero.SetInfoCallback(AHandler, AUserData);
  end;
end;

function  jiGetLoadModelProgressCallback(): jiLoadModelProgressCallback;
begin
  Result := nil;

  if Assigned(LJetInfero) then
  begin
    Result := LJetInfero.GetLoadModelProgressCallback();
  end;
end;

procedure jiSetLoadModelProgressCallback(const AHandler: jiLoadModelProgressCallback; const AUserData: Pointer);
begin
  if Assigned(LJetInfero) then
  begin
    LJetInfero.SetLoadModelProgressCallback(AHandler, AUserData);
  end;
end;

function  jiGetLoadModelCallback(): jiLoadModelCallback;
begin
  Result := nil;

  if Assigned(LJetInfero) then
  begin
    Result := LJetInfero.GetLoadModelCallback();
  end;
end;

procedure jiSetLoadModelCallback(const AHandler: jiLoadModelCallback; const AUserData: Pointer);
begin
  if Assigned(LJetInfero) then
  begin
    LJetInfero.SetLoadModelCallback(AHandler, AUserData);
  end;
end;

function  jiGetInferenceStartCallback(): jiInferenceStartCallback;
begin
  Result := nil;

  if Assigned(LJetInfero) then
  begin
    Result := LJetInfero.GetInferenceStartCallback();
  end;
end;

procedure jiSetInferenceStartCallback(const AHandler: jiInferenceStartCallback; const AUserData: Pointer);
begin
  if Assigned(LJetInfero) then
  begin
    LJetInfero.SetInferenceStartCallback(AHandler, AUserData);
  end;
end;

function  jiGetInferenceEndCallback(): jiInferenceEndCallback;
begin
  Result := nil;

  if Assigned(LJetInfero) then
  begin
    Result := LJetInfero.GetInferenceEndCallback();
  end;
end;

procedure jiSetInferenceEndCallback(const AHandler: jiInferenceEndCallback; const AUserData: Pointer);
begin
  if Assigned(LJetInfero) then
  begin
    LJetInfero.SetInferenceEndCallback(AHandler, AUserData);
  end;
end;

procedure jiSetTokenRightMargin(const AMargin: Int32);
begin
  if not Assigned(LJetInfero) then Exit;
  LJetInfero.SetTokenRightMargin(AMargin);
end;

procedure jiSetTokenMaxLineLength(const ALength: Int32);
begin
  if not Assigned(LJetInfero) then Exit;
  LJetInfero.SetTokenMaxLineLength(ALength);
end;

function  jiSaveConfig(const AFilename: PWideChar): Boolean;
begin
  Result := False;
  if not Assigned(LJetInfero) then Exit;

  Result := LJetInfero.SaveConfig(string(AFilename));
end;

function  jiLoadConfig(const AFilename: PWideChar): Boolean;
begin
  Result := False;
  if not Assigned(LJetInfero) then Exit;

  Result := LJetInfero.LoadConfig(string(AFilename));
end;

procedure jiClearModelDefines();
begin
  if not Assigned(LJetInfero) then Exit;

  LJetInfero.ClearModelDefines();
end;

function  jiDefineModel(const AFilename, ARefName, ATemplate, ATemplateEnd: PWideChar; ACapitalizeRole: Boolean; const AMaxContext: UInt32; const AMainGPU: Int32; const AGPULayers: Int32; const AMaxThreads: Int32): Boolean;
begin
  Result := False;
  if not Assigned(LJetInfero) then Exit;

  Result := LJetInfero.DefineModel(AFilename, string(ARefName), string(ATemplate), string(ATemplateEnd), ACapitalizeRole, AMaxContext, AMainGPU, AGPULayers, AMaxThreads)
end;

function  jiRemoveModelDefine(const ARefName: PWideChar): Boolean;
begin
  Result := False;
  if not Assigned(LJetInfero) then Exit;

  Result := LJetInfero.RemoveModelDefine(ARefName);
end;

function jiSaveModelDefines(const AFilename: PWideChar): Boolean;
begin
  Result := False;
  if not Assigned(LJetInfero) then Exit;

  Result := LJetInfero.SaveModelDefines(AFilename);
end;

function  jiLoadModelDefines(const AFilename: PWideChar): Boolean;
begin
  Result := False;
  if not Assigned(LJetInfero) then Exit;

  Result := LJetInfero.LoadModelDefines(AFilename);
end;

function  jiLoadModel(const AModelRef: PWideChar): Boolean;
begin
  Result := False;
  if not Assigned(LJetInfero) then Exit;

  Result := LJetInfero.LoadModel(AModelRef);
end;

procedure jiUnloadModel();
begin
  if not Assigned(LJetInfero) then Exit;

  LJetInfero.UnloadModel();
end;

procedure jiClearMessages();
begin
  if not Assigned(LJetInfero) then Exit;

  LJetInfero.ClearMessages();
end;

function  jiAddMessage(const ARole, AContent: PWideChar): Int32;
begin
  Result := 0;
  if not Assigned(LJetInfero) then Exit;

  Result := LJetInfero.AddMessage(ARole, AContent);
end;

function  jiGetLastUserMessage(): PWideChar;
begin
  Result := nil;
  if not Assigned(LJetInfero) then Exit;

  Result := PWideChar(LJetInfero.GetLastUserMessage());
end;

function  jiGetMessagePrompt(const AModelRef: PWideChar): PWideChar;
begin
  Result := nil;
  if not Assigned(LJetInfero) then Exit;

  Result := PWideChar(LJetInfero.GetMessagePrompt(AModelRef));
end;

function  jiRunInference(const AModelRef: PWideChar): Boolean;
begin
  Result := False;
  if not Assigned(LJetInfero) then Exit;

  Result := LJetInfero.RunInference(string(AModelRef));
end;

function  jiGetInferenceResponse(): PWideChar;
begin
  Result := nil;
  if not Assigned(LJetInfero) then Exit;

  Result := PWideChar(LJetInfero.GetInferenceResponse());
end;

procedure jiGetPerformanceResult(ATokensPerSecond: PDouble; ATotalInputTokens: PInt32; ATotalOutputTokens: PInt32);
var
  LPerf: TJetInfero.PerformanceResult;
begin
  LPerf := Default(TJetInfero.PerformanceResult);
  if not Assigned(LJetInfero) then Exit;

  LPerf := LJetInfero.GetPerformanceResult();

  if Assigned(ATokensPerSecond) then
    ATokensPerSecond^ := LPerf.TokensPerSecond;

  if Assigned(ATotalInputTokens) then
    ATotalInputTokens^ := LPerf.TotalInputTokens;

  if Assigned(ATotalOutputTokens) then
    ATotalOutputTokens^ := LPerf.TotalOutputTokens;
end;

{ TModel }
function TModel.ToJSON(): TJSONObject;
begin
  Result := TJSONObject.Create;
  Result.AddPair('Filename', Filename);
  Result.AddPair('RefName', RefName);
  Result.AddPair('MaxContext', TJSONNumber.Create(MaxContext));
  Result.AddPair('MaxThreads', TJSONNumber.Create(MaxThreads));
  Result.AddPair('MainGPU', TJSONNumber.Create(MainGPU));
  Result.AddPair('GPULayers', TJSONNumber.Create(GPULayers));
  Result.AddPair('Template', Template);
  Result.AddPair('TemplateEnd', TemplateEnd);
end;

procedure TModel.FromJSON(const AJSON: TJSONObject);
begin
  Filename := AJSON.GetValue<string>('Filename', '');
  RefName := AJSON.GetValue<string>('RefName', '');
  MaxContext := AJSON.GetValue<UInt32>('MaxContext', 0);
  MaxThreads := AJSON.GetValue<Int32>('MaxThreads', 0);
  MainGPU := AJSON.GetValue<Int32>('MainGPU', 0);
  GPULayers := AJSON.GetValue<Int32>('GPULayers', 0);
  Template := AJSON.GetValue<string>('Template', '');
  TemplateEnd := AJSON.GetValue<string>('TemplateEnd', '');
end;


{ TJetInfero }
function TJetInfero.GetClibsDLLVersion(): string;
var
  Size, Handle: DWORD;
  Buffer: Pointer;
  FileInfo: Pointer;
  VerValue: PVSFixedFileInfo;
  VerSize: UINT;
  LFilename: string;

begin
  Result := '0.0.0'; // Default version if info not found

  LFilename := GetCurrentDLLFilename();

  // Get the size of the version info block
  Size := GetFileVersionInfoSize(PChar(LFilename), Handle);
  if Size = 0 then
    Exit;

  // Allocate buffer for version info
  GetMem(Buffer, Size);
  try
    // Retrieve version info
    if GetFileVersionInfo(PChar(LFilename), Handle, Size, Buffer) then
    begin
      // Query version info
      if VerQueryValue(Buffer, '\', FileInfo, VerSize) and (VerSize >= SizeOf(TVSFixedFileInfo)) then
      begin
        VerValue := PVSFixedFileInfo(FileInfo);
        Result := Format('%d.%d.%d',
          [HiWord(VerValue.dwFileVersionMS), LoWord(VerValue.dwFileVersionMS),
           HiWord(VerValue.dwFileVersionLS)]);
      end;
    end;
  finally
    FreeMem(Buffer);
  end;
end;

procedure TJetInfero.UnloadDLL();
begin
  // unload deps DLL
  if FCLibsHandle <> 0 then
  begin
    FreeLibrary(FCLibsHandle);
    TFile.Delete(FClibsDLLFilename);
    FCLibsHandle := 0;
    FClibsDLLFilename := '';
  end;
end;

function TJetInfero.LoadDLL(): Boolean;
var
  LResStream: TResourceStream;

  function d55a860b2915413c84d3620b9cbee959(): string;
  const
    CValue = 'b87deef5bbfd43c3a07379e26f4dec9b';
  begin
    Result := CValue;
  end;

begin
  Result := False;
  FLastError := '';

  // load deps DLL
  if FCLibsHandle <> 0 then Exit;
  if not Boolean((FindResource(HInstance, PChar(d55a860b2915413c84d3620b9cbee959()), RT_RCDATA) <> 0)) then
  begin
    SetError('Failed to find Deps DLL resource', []);
    Exit;
  end;
  LResStream := TResourceStream.Create(HInstance, d55a860b2915413c84d3620b9cbee959(), RT_RCDATA);
  try
    LResStream.Position := 0;
    FClibsDLLFilename := TPath.Combine(TPath.GetTempPath, TPath.ChangeExtension(TPath.GetGUIDFileName.ToLower, '.'));
    LResStream.SaveToFile(FClibsDLLFilename);
    if not TFile.Exists(FClibsDLLFilename) then
    begin
      SetError('Failed to find extracted Deps DLL', []);
      Exit;
    end;

    FCLibsHandle := LoadLibrary(PChar(FClibsDLLFilename));
    if FCLibsHandle = 0 then
    begin
      SetError('Failed to load extracted Deps DLL', []);
      Exit;
    end;

    Result := True;
  finally
    LResStream.Free();
  end;
  GetExports(FCLibsHandle);
end;

function TJetInfero.TokenToPiece(const AVocab: Pllama_vocab; const AContext: Pllama_context; const AToken: llama_token; const ASpecial: Boolean): string;
var
  LTokens: Int32;
  LCheck: Int32;
  LBuffer: TArray<UTF8Char>;
begin
  try
    SetLength(LBuffer, 9);
    LTokens := llama_token_to_piece(AVocab, AToken, @LBuffer[0], 8, 0, ASpecial);
    if LTokens < 0 then
      begin
        SetLength(LBuffer, (-LTokens)+1);
        LCheck := llama_token_to_piece(AVocab, AToken, @LBuffer[0], -LTokens, 0, ASpecial);
        Assert(LCheck = -LTokens);
        LBuffer[-LTokens] := #0;
      end
    else
      begin
        LBuffer[LTokens] := #0;
      end;
    Result := UTF8ToString(@LBuffer[0]);
  except
    on E: Exception do
    begin
      SetError(E.Message, []);
      Exit;
    end;
  end;
end;


function TJetInfero.CalcPerformance(const AContext: Pllama_context): PerformanceResult;
var
  LTotalTimeSec: Double;
  APerfData: llama_perf_context_data;
begin
  APerfData := llama_perf_context(AContext);

  // Convert milliseconds to seconds
  LTotalTimeSec := APerfData.t_eval_ms / 1000;

  // Total input tokens (n_p_eval assumed to be input tokens)
  Result.TotalInputTokens := APerfData.n_p_eval;

  // Total output tokens (n_eval assumed to be output tokens)
  Result.TotalOutputTokens := APerfData.n_eval;

  // Calculate tokens per second (total tokens / time in seconds)
  if LTotalTimeSec > 0 then
    Result.TokensPerSecond := (Result.TotalInputTokens + Result.TotalOutputTokens) / LTotalTimeSec
  else
    Result.TokensPerSecond := 0;
end;

procedure TJetInfero.OnNextToken(const AToken: string);
begin
  FInference.Response := FInference.Response + AToken;

  if Assigned(FCallbacks.InferenceToken.Handler) then
    begin
      FCallbacks.InferenceToken.Handler(PWideChar(AToken), FCallbacks.InferenceToken.UserData);
    end
  else
    begin
      if HasConsoleOutput() then
      begin
        Write(AToken);
      end;
    end;
end;

function  TJetInfero.OnInferenceCancel(): Boolean;
begin
  if Assigned(FCallbacks.InferenceCancel.Handler) then
    begin
      Result := FCallbacks.InferenceCancel.Handler(FCallbacks.InferenceCancel.UserData);
    end
  else
    begin
      Result := Boolean(GetAsyncKeyState(VK_ESCAPE) <> 0);
    end;
end;

procedure TJetInfero.OnInfo(const ALevel: Integer; const AText: string);
begin
  if Assigned(FCallbacks.Info.Handler) then
    begin
      FCallbacks.Info.Handler(ALevel, PWideChar(AText), FCallbacks.Info.UserData);
    end
  else
    begin
      if HasConsoleOutput() then
      begin
        Write(AText);
      end;
    end;
end;

function  TJetInfero.OnLoadModelProgress(const AModelRef: string; const AProgress: Single): Boolean;
begin
  Result := True;

  if Assigned(FCallbacks.LoadModelProgress.Handler) then
    begin
      Result := FCallbacks.LoadModelProgress.Handler(PWideChar(AModelRef), AProgress, FCallbacks.LoadModelProgress.UserData);
    end
  else
    begin
      if HasConsoleOutput() then
      begin
        Write(Format(#13+'Loading model "%s" (%3.2f%s)...', [AModelRef, AProgress*100, '%']));
        if AProgress >= 1 then
        begin
          Write(#27'[2K'); // Clear the current line
          Write(#27'[G');  // Move cursor to the beginning of the line
        end;
      end;
    end;
end;

procedure TJetInfero.OnLoadModel(const AModelRef: string; const ASuccess: Boolean);
begin
  if Assigned(FCallbacks.LoadModel.Handler) then
  begin
    FCallbacks.LoadModel.Handler(PWideChar(AModelRef), ASuccess, FCallbacks.LoadModel.UserData);
  end;
end;

procedure TJetInfero.OnInferenceStart();
begin
  if Assigned(FCallbacks.InferenceStart.Handler) then
  begin
    FCallbacks.InferenceStart.Handler(FCallbacks.InferenceStart.UserData);
  end;
end;

procedure TJetInfero.OnInferenceEnd();
begin
  if Assigned(FCallbacks.InferenceEnd.Handler) then
  begin
    FCallbacks.InferenceEnd.Handler(FCallbacks.InferenceEnd.UserData);
  end;
end;

constructor TJetInfero.Create();
begin
  inherited;
  FMessages := TMessages.Create();
  FModels:= TModels.Create();
end;

destructor TJetInfero.Destroy();
begin
  UnloadModel();

  if Assigned(FModels) then
    FModels.Free();

  if Assigned(FMessages) then
    FMessages.Free();

  Shutdown();
  inherited;
end;

function  TJetInfero.Startup(): Boolean;
begin
  Result := LoadDLL();
  if not Result then Exit;
  FVersion := GetClibsDLLVersion();

  FTokenMaxLineLength := FTokenResponse.MaxLineLength;
  FTokenRightMargin :=  FTokenResponse.RightMargin;
end;

procedure TJetInfero.Shutdown();
begin
  UnloadDLL();
end;

procedure TJetInfero.SetError(const AText: string; const AArgs: array of const);
begin
  FLastError := Format(AText, AArgs);
end;

function  TJetInfero.GetLastError(): string;
begin
  Result := FLastError;
end;

function  TJetInfero.GetVersion(): string;
begin
  Result := FVersion;
end;

function  TJetInfero.GetInferenceCancelCallback(): jiInferenceCancelCallback;
begin
  Result := FCallbacks.InferenceCancel.Handler;
end;

procedure TJetInfero.SetInferenceCancelCallback(const AHandler: jiInferenceCancelCallback; const AUserData: Pointer);
begin
  FCallbacks.InferenceCancel.Handler := AHandler;
  FCallbacks.InferenceCancel.UserData := AUserData;
end;

function  TJetInfero.GetInferenceTokenCallback(): jiInferenceTokenCallback;
begin
  Result := FCallbacks.InferenceToken.Handler;
end;

procedure TJetInfero.SetInferenceTokenlCallback(const AHandler: jiInferenceTokenCallback; const AUserData: Pointer);
begin
  FCallbacks.InferenceToken.Handler := AHandler;
  FCallbacks.InferenceToken.UserData := AUserData;
end;

function  TJetInfero.GetInfoCallback(): jiInfoCallback;
begin
  Result := FCallbacks.Info.Handler;
end;

procedure TJetInfero.SetInfoCallback(const AHandler: jiInfoCallback; const AUserData: Pointer);
begin
  FCallbacks.Info.Handler := AHandler;
  FCallbacks.Info.UserData := AUserData;
end;

function  TJetInfero.GetLoadModelProgressCallback(): jiLoadModelProgressCallback;
begin
  Result := FCallbacks.LoadModelProgress.Handler;
end;

procedure TJetInfero.SetLoadModelProgressCallback(const AHandler: jiLoadModelProgressCallback; const AUserData: Pointer);
begin
  FCallbacks.LoadModelProgress.Handler := AHandler;
  FCallbacks.LoadModelProgress.UserData := AUserData;
end;

function  TJetInfero.GetLoadModelCallback(): jiLoadModelCallback;
begin
  Result := FCallbacks.LoadModel.Handler;
end;

procedure TJetInfero.SetLoadModelCallback(const AHandler: jiLoadModelCallback; const AUserData: Pointer);
begin
  FCallbacks.LoadModel.Handler := AHandler;
  FCallbacks.LoadModel.UserData := AUserData;
end;

function  TJetInfero.GetInferenceStartCallback(): jiInferenceStartCallback;
begin
  Result := FCallbacks.InferenceStart.Handler;
end;

procedure TJetInfero.SetInferenceStartCallback(const AHandler: jiInferenceStartCallback; const AUserData: Pointer);
begin
  FCallbacks.InferenceStart.Handler := AHandler;
  FCallbacks.InferenceStart.UserData := AUserData;
end;

function  TJetInfero.GetInferenceEndCallback(): jiInferenceEndCallback;
begin
  Result := FCallbacks.InferenceEnd.Handler;
end;

procedure TJetInfero.SetInferenceEndCallback(const AHandler: jiInferenceEndCallback; const AUserData: Pointer);
begin
  FCallbacks.InferenceEnd.Handler := AHandler;
  FCallbacks.InferenceEnd.UserData := AUserData;
end;

procedure TJetInfero.SetTokenRightMargin(const AMargin: Int32);
begin
  FTokenRightMargin := EnsureRange(AMargin, 0, MaxInt);
  FTokenResponse.SetRightMargin(AMargin);
end;

procedure TJetInfero.SetTokenMaxLineLength(const ALength: Int32);
begin
  FTokenMaxLineLength := EnsureRange(ALength, 0, MaxInt);
  FTokenResponse.SetMaxLineLength(ALength);
end;

function TJetInfero.SaveConfig(const AFilename: string): Boolean;
var
  LIniFile: TIniFile;
  LFilename: string;
begin
  LFilename := TPath.GetFullPath(TPath.ChangeExtension(AFilename, 'ini'));

  try
    LIniFile := TIniFile.Create(LFilename);
    try
      // Save the configuration values to the INI file
      LIniFile.WriteInteger('Config', 'FTokenRightMargin', FTokenRightMargin);
      LIniFile.WriteInteger('Config', 'FTokenMaxLineLength', FTokenMaxLineLength);
      Result := True;
    finally
      LIniFile.Free;
    end;
  except
    // Handle exceptions if needed (e.g., log the error)
    Result := False;
  end;
end;

function TJetInfero.LoadConfig(const AFilename: string): Boolean;
var
  LIniFile: TIniFile;
  LFilename: string;
begin
  Result := False;
  if not FileExists(AFilename) then
    Exit; // File does not exist

  LFilename := TPath.GetFullPath(TPath.ChangeExtension(AFilename, 'ini'));

  try
    LIniFile := TIniFile.Create(LFilename);
    try
      // Load the configuration values from the INI file
      FTokenRightMargin := LIniFile.ReadInteger('Config', 'FTokenRightMargin', 10);
      FTokenMaxLineLength := LIniFile.ReadInteger('Config', 'FTokenMaxLineLength', 120);
      Result := True;
    finally
      LIniFile.Free;
    end;
  except
    // Handle exceptions if needed (e.g., log the error)
    Result := False;
  end;
end;

procedure TJetInfero.ClearModelDefines();
begin
  FModels.Clear();
end;

function  TJetInfero.DefineModel(const AFilename, ARefName, ATemplate, ATemplateEnd: string; ACapitalizeRole: Boolean; const AMaxContext: UInt32; const AMainGPU: Int32; const AGPULayers: Int32; const AMaxThreads: Int32): Boolean;
var
  LModel: TModel;
begin
  LModel.Filename := AFilename;
  LModel.RefName := ARefName;
  LModel.MaxContext := AMaxContext;
  LModel.MainGPU := AMainGPU;
  LModel.GPULayers := AGPULayers;
  LModel.Template := ATemplate;
  LModel.TEmplateEnd := ATemplateEnd;
  LModel.MaxThreads := AMaxThreads;
  LModel.CapitalizeRole := ACapitalizeRole;

  Result := FModels.TryAdd(ARefName, LModel);
end;

function  TJetInfero.RemoveModelDefine(const ARefName: string): Boolean;
begin
  Result := False;

  if FModels.ContainsKey(ARefName) then
  begin
    FModels.Remove(ARefName);
    Result := True;
  end;
end;

function TJetInfero.SaveModelDefines(const AFilename: string): Boolean;
var
  JSONRoot: TJSONObject;
  ModelPair: TPair<string, TModel>;
  ModelObj: TJSONObject;
  ModelsArray: TJSONArray;
begin
  JSONRoot := TJSONObject.Create;
  try
    ModelsArray := TJSONArray.Create;
    for ModelPair in FModels do
    begin
      ModelObj := ModelPair.Value.ToJSON;
      //ModelObj.AddPair('Key', ModelPair.Key);
      ModelsArray.AddElement(ModelObj);
    end;
    JSONRoot.AddPair('Models', ModelsArray);

    TFile.WriteAllText(AFilename, JSONRoot.Format());

    Result := TFile.Exists(AFilename);
  finally
    JSONRoot.Free;
  end;
end;

function TJetInfero.LoadModelDefines(const AFilename: string): Boolean;
var
  JSONRoot: TJSONObject;
  ModelsArray: TJSONArray;
  ModelObj: TJSONObject;
  Model: TModel;
  Key: string;
  JSONText: string;
  I: Integer;
begin
  Result := False;
  if not TFile.Exists(AFilename) then Exit;

  ClearModelDefines();

  JSONText := TFile.ReadAllText(AFilename);
  JSONRoot := TJSONObject.ParseJSONValue(JSONText) as TJSONObject;
  try
    ModelsArray := JSONRoot.GetValue<TJSONArray>('Models');
    for I := 0 to ModelsArray.Count - 1 do
    begin
      ModelObj := ModelsArray.Items[I] as TJSONObject;
      Key := ModelObj.GetValue<string>('RefName');
      Model.FromJSON(ModelObj);
      FModels.Add(Key, Model);
    end;

    Result := True;
  finally
    JSONRoot.Free;
  end;
end;

procedure TJetInfero_CErrCallback(const AText: PUTF8Char; AUserData: Pointer); cdecl;
begin
  if Assigned(AUserData) then
    TJetInfero(AUserData).OnInfo(GGML_LOG_LEVEL_ERROR, Utf8ToString(AText));
end;

procedure TJetInfero_LogCallback(ALevel: ggml_log_level; const AText: PUTF8Char; AUserData: Pointer); cdecl;
begin
  if Assigned(AUserData) then
    TJetInfero(AUserData).OnInfo(ALevel, Utf8ToString(AText));
end;

function TJetInfero_ProgressCallback(AProgress: single; AUserData: pointer): Boolean; cdecl;
var
  LJetInfero: TJetInfero;
begin
  LJetInfero := AUserData;
  if Assigned(LJetInfero) then
    Result := LJetInfero.OnLoadModelProgress(LJetInfero.FInference.Model.Filename, AProgress)
  else
    Result := True;
end;


function  TJetInfero.LoadModel(const AModelRef: string): Boolean;
var
  LModelParams: llama_model_params;
begin
  Result := False;

  if SameText(FInference.Model.RefName, AModelRef) then
  begin
    Result := True;
    Exit;
  end;

  if not FModels.TryGetValue(AModelRef, FInference.Model) then Exit;

  redirect_cerr_to_callback(TJetInfero_CerrCallback, nil);

  llama_log_set(TJetInfero_LogCallback, Self);

  LModelParams := llama_model_default_params();

  LModelParams.progress_callback := TJetInfero_ProgressCallback;
  LModelParams.progress_callback_user_data := Self;
  LModelParams.main_gpu := FInference.Model.MainGPU;

  if FInference.Model.GPULayers < 0 then
    LModelParams.n_gpu_layers := MaxInt
  else
    LModelParams.n_gpu_layers := FInference.Model.GPULayers;

  FModel :=  llama_load_model_from_file(AsUtf8(FInference.Model.Filename), LModelParams);
  if not Assigned(FModel) then
  begin
    OnLoadModel(FInference.Model.Filename, False);
    SetError('Failed to load model: "%s"', [FInference.Model.Filename]);
    Exit;
  end;
  OnLoadModel(FInference.Model.Filename, True);

  FInference.Active := False;
  FInference.Prompt := '';
  FInference.Response := '';

  Result := True;
end;

procedure TJetInfero.UnloadModel();
begin
  if Assigned(FModel) then
  begin
    llama_free_model(FModel);
    FModel := nil;
    restore_cerr();
    FInference := Default(TInference);
  end;
end;

procedure TJetInfero.ClearMessages();
begin
  FMessages.Clear();
  FLastUserMessage := '';
end;

function TJetInfero.AddMessage(const ARole, AContent: string): Int32;
var
  LMessage: TMessage;
begin
  LMessage.Role := ARole;
  LMessage.Content := AContent;
  FMessages.Add(LMessage);
  Result := FMessages.Count;
  if ContainsText(ARole, 'user') then
    FLastUserMessage := AContent;
end;

function  TJetInfero.GetLastUserMessage(): string;
begin
  Result := FLastUserMessage;
end;

function  TJetInfero.GetMessagePrompt(const AModelRef: string): string;
var
  LModel: TModel;
  LItem: TMessage;
  LMessage: TMessage;
begin
  Result := '';
  FInference.Prompt := '';

  if FModels.TryGetValue(AModelRef, LModel) then
  begin
    for LItem in FMessages do
    begin
      LMessage := LItem;
      if LModel.CapitalizeRole then
        LMessage.Role := CapitalizeFirstChar(LMessage.Role);
      FInference.Prompt := FInference.Prompt + LModel.Template.Replace('{role}', LMessage.Role).Replace('{content}', LMessage.Content).Trim;
    end;
    FInference.Prompt := FInference.Prompt + LModel.TemplateEnd;
  end;

  Result := FInference.Prompt;
end;

function  TJetInfero.RunInference(const AModelRef: string): Boolean;
var
  LNumPrompt: Integer;
  LPromptTokens: TArray<llama_token>;
  LCtxParams: llama_context_params;
  LNumPredict: integer;
  LCtx: Pllama_context;
  LSmplrParams: llama_sampler_chain_params;
  LSmplr: Pllama_sampler;
  N: Integer;
  LTokenStr: string;
  LBatch: llama_batch;
  LNewTokenId: llama_token;
  LNumPos: Integer;
  LPrompt: UTF8String;
  LFirstToken: Boolean;
  V: Int32;
  LBuf: array[0..255] of UTF8Char;
  LKey: string;
  LMaxContext: integer;
  LVocab: Pllama_vocab;
begin
  Result := False;

  // check if inference is already runnig
  if FInference.Active then
  begin
    SetError('[%s] Inference already active', ['RunInference']);
    Exit;
  end;

  // start new inference
  FInference := Default(TInference);

  // check if model not loaded
  if not LoadModel(AModelRef) then
  begin
    Exit;
  end;

  FInference.Prompt := GetMessagePrompt(AModelRef);
  if FInference.Prompt.IsEmpty then
  begin
    SetError('[%s] Inference prompt was empty', ['RunInference']);
    Exit;
  end;

  FInference.Active := True;
  FInference.Response := '';

  FLastError := '';
  LFirstToken := True;
  LMaxContext := 0;

  for V := 0 to llama_model_meta_count(FModel)-1 do
  begin
    llama_model_meta_key_by_index(FModel, V, @LBuf[0], length(LBuf));
    LKey := string(LBuf);
    if LKey.Contains('context_length') then
    begin
      llama_model_meta_val_str_by_index(FModel, V, @LBuf[0], length(LBuf));
      LKey := string(LBuf);
      LMaxContext :=  LKey.ToInteger;
      break;
    end;
  end;

  if LMaxContext > 0 then
    LNumPredict := EnsureRange(FInference.Model.MaxContext, 512, LMaxContext)
  else
    LNumPredict := 512;

  LVocab := llama_model_get_vocab(FModel);

  LPrompt := UTF8String(FInference.Prompt);

  LNumPrompt := -llama_tokenize(LVocab, PUTF8Char(LPrompt), Length(LPrompt), nil, 0, true, true);

  SetLength(LPromptTokens, LNumPrompt);

  if llama_tokenize(LVocab, PUTF8Char(LPrompt), Length(LPrompt), @LPromptTokens[0], Length(LPromptTokens), true, true) < 0 then
  begin
    SetError('Failed to tokenize prompt', []);
  end;

  LCtxParams := llama_context_default_params();
  LCtxParams.n_ctx := LNumPrompt + LNumPredict - 1;
  LCtxParams.n_batch := LNumPrompt;
  LCtxParams.no_perf := false;
  LCtxParams.n_threads := EnsureRange(FInference.Model.MaxThreads, 1, GetPhysicalProcessorCount());
  LCtxParams.n_threads_batch := LCtxParams.n_threads;

  LCtx := llama_new_context_with_model(FModel, LCtxParams);
  if LCtx = nil then
  begin
    SetError('Failed to create inference context', []);
    llama_free_model(FModel);
    exit;
  end;

  LSmplrParams := llama_sampler_chain_default_params();
  LSmplr := llama_sampler_chain_init(LSmplrParams);
  llama_sampler_chain_add(LSmplr, llama_sampler_init_greedy());

  LBatch := llama_batch_get_one(@LPromptTokens[0], Length(LPromptTokens));

  LNumPos := 0;

  FInference.Perf := Default(TJetInfero.PerformanceResult);

  OnInferenceStart();
  while LNumPos + LBatch.n_tokens < LNumPrompt + LNumPredict do
  begin
    if OnInferenceCancel() then Break;

    N := llama_decode(LCtx, LBatch);
    if N <> 0 then
    begin
      SetError('Failed to decode context', []);
      llama_sampler_free(LSmplr);
      llama_free(LCtx);
      llama_free_model(FModel);
      Exit;
    end;

    LNumPos := LNumPos + LBatch.n_tokens;

    LNewTokenId := llama_sampler_sample(LSmplr, LCtx, -1);

    if llama_vocab_is_eog(LVocab, LNewTokenId) then
    begin
      break;
    end;

    LTokenStr := TokenToPiece(LVocab, LCtx, LNewTokenId, false);

    if LFirstToken then
    begin
      LTokenStr := LTokenStr.Trim();
      LFirstToken := False;
    end;

    case FTokenResponse.AddToken(LTokenStr) of
      tpaWait:
      begin
      end;

      tpaAppend:
      begin
        OnNextToken(FTokenResponse.LastWord(False));
      end;

      tpaNewline:
      begin
        OnNextToken(#10);
        OnNextToken(FTokenResponse.LastWord(True));
      end;
    end;

    LBatch := llama_batch_get_one(@LNewTokenId, 1);
  end;

  if FTokenResponse.Finalize then
  begin
    case FTokenResponse.AddToken('') of
      tpaWait:
      begin
      end;

      tpaAppend:
      begin
        OnNextToken(FTokenResponse.LastWord(False));
      end;

      tpaNewline:
      begin
        OnNextToken(#10);
        OnNextToken(FTokenResponse.LastWord(True));
      end;
    end;
  end;

  OnInferenceEnd();

  FInference.Perf := CalcPerformance(LCtx);

  llama_sampler_free(LSmplr);
  llama_free(LCtx);

  Result := True;
end;

function  TJetInfero.GetInferenceResponse(): string;
begin
  Result := FInference.Response;
end;

function  TJetInfero.GetPerformanceResult(): TJetInfero.PerformanceResult;
begin
  Result := FInference.Perf;
end;

initialization
begin
  ReportMemoryLeaksOnShutdown := True;

  SetConsoleCP(CP_UTF8);
  SetConsoleOutputCP(CP_UTF8);
  EnableVirtualTerminalProcessing();
end;

finalization
begin
end;

end.
