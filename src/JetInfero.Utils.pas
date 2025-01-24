unit JetInfero.Utils;

interface

uses
  WinApi.Windows,
  System.SysUtils,
  System.StrUtils;

function  GetCurrentDLLFilename(): string;
procedure GetConsoleSize(AWidth: PInteger; AHeight: PInteger);
function  AsUTF8(const AText: string): Pointer;
function  ContainsText(const AText, ASubText: string): Boolean;
function  CapitalizeFirstChar(const AText: string): string;
function  GetPhysicalProcessorCount(): DWORD;
function  SanitizeToJson(const aText: string): string;
function  SanitizeFromJson(const aText: string): string;
function  HasConsoleOutput(): Boolean;
function  EnableVirtualTerminalProcessing(): DWORD;

type
  { TCallback<T> }
  TCallback<T> = record
    Handler: T;
    UserData: Pointer;
  end;

  { TTokenResponse }

  // AddToken return messages - for TResponse.AddToken
  //  paWait = No new (full) words, just wait for more incoming tokens
  //  Append = Append existing line with latest word
  //  NewLine = start new line then print the latest word
  TTokenPrintAction = (tpaWait, tpaAppend, tpaNewline);

  { TResponse
    Helper to handle incoming tokens during streaming
      Example uses:
      - Tabulate tokens into full words based on wordbreaks
      - Control wordwrap/linechanges for console or custom GUI without wordwrap functionality
        (Does change the print resolution from Token to logical words)
  }
  TTokenResponse = record
  private
    FRaw: string;                  // Full response as is
    FTokens: array of string;      // Actual tokens
    FMaxLineLength: Integer;       // Define confined space, in chars for fixed width font
    FWordBreaks: array of char;    // What is considered a logical word-break
    FLineBreaks: array of char;    // What is considered a logical line-break
    FWords: array of String;       // Response but as array of "words"
    FWord: string;                // Current word accumulating
    FLine: string;                // Current line accumulating
    FFinalized: Boolean;          // Know the finalization is done
    FRightMargin: Integer;
    function HandleLineBreaks(const AToken: string): Boolean;
    function SplitWord(const AWord: string; var APrefix, ASuffix: string): Boolean;
    function GetLineLengthMax(): Integer;
  public
    class operator Initialize (out ADest: TTokenResponse);
    property RightMargin: Integer read FRightMargin;
    property MaxLineLength: Integer read FMaxLineLength;
    procedure SetRightMargin(const AMargin: Integer);
    procedure SetMaxLineLength(const ALength: Integer);
    function AddToken(const aToken: string): TTokenPrintAction;
    function LastWord(const ATrimLeft: Boolean=False): string;
    function Finalize: Boolean;
  end;

implementation

var
  FMarshaller: TMarshaller;

function GetModuleHandleEx(dwFlags: DWORD; lpModuleName: LPCSTR; var phModule: HMODULE): BOOL; stdcall; external 'kernel32.dll' name 'GetModuleHandleExA';

function GetCurrentDLLFilename(): string;
const
  GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS = $00000004;
  GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT = $00000002;
var
  ModuleName: array[0..MAX_PATH] of Char;
  ModuleHandle: HMODULE;

begin
  ModuleHandle := 0;
  if GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS or GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, @GetCurrentDLLFilename, ModuleHandle) then
  begin
    if GetModuleFileName(ModuleHandle, ModuleName, SizeOf(ModuleName)) > 0 then
      Result := ModuleName
    else
      Result := '';
    FreeLibrary(ModuleHandle);  // Decrement the reference count
  end
  else
    Result := '';
end;

procedure GetConsoleSize(AWidth: PInteger; AHeight: PInteger);
var
  LConsoleInfo: TConsoleScreenBufferInfo;
begin
  GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), LConsoleInfo);
  if Assigned(AWidth) then
    AWidth^ := LConsoleInfo.dwSize.X;

  if Assigned(AHeight) then
  AHeight^ := LConsoleInfo.dwSize.Y;
end;

function AsUTF8(const AText: string): Pointer;
begin
  Result := FMarshaller.AsUtf8(AText).ToPointer;
end;

function ContainsText(const AText, ASubText: string): Boolean;
begin
  Result := Pos(UpperCase(ASubText), UpperCase(AText)) > 0;
end;

function CapitalizeFirstChar(const AText: string): string;
begin
  if AText = '' then
    Exit(AText); // Return an empty string if the input is empty
  Result := UpperCase(AText[1]) + Copy(AText, 2, Length(AText) - 1);
end;

function GetPhysicalProcessorCount(): DWORD;
var
  BufferSize: DWORD;
  Buffer: PSYSTEM_LOGICAL_PROCESSOR_INFORMATION;
  ProcessorInfo: PSYSTEM_LOGICAL_PROCESSOR_INFORMATION;
  Offset: DWORD;
begin
  Result := 0;
  BufferSize := 0;

  // Call GetLogicalProcessorInformation with buffer size set to 0 to get required buffer size
  if not GetLogicalProcessorInformation(nil, BufferSize) and (WinApi.Windows.GetLastError() = ERROR_INSUFFICIENT_BUFFER) then
  begin
    // Allocate buffer
    GetMem(Buffer, BufferSize);
    try
      // Call GetLogicalProcessorInformation again with allocated buffer
      if GetLogicalProcessorInformation(Buffer, BufferSize) then
      begin
        ProcessorInfo := Buffer;
        Offset := 0;

        // Loop through processor information to count physical processors
        while Offset + SizeOf(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= BufferSize do
        begin
          if ProcessorInfo.Relationship = RelationProcessorCore then
            Inc(Result);

          Inc(ProcessorInfo);
          Inc(Offset, SizeOf(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
        end;
      end;
    finally
      FreeMem(Buffer);
    end;
  end;
end;

function  SanitizeToJson(const aText: string): string;
var
  i: Integer;
begin
  Result := '';
  for i := 1 to Length(aText) do
  begin
    case aText[i] of
      '\': Result := Result + '\\';
      '"': Result := Result + '\"';
      '/': Result := Result + '\/';
      #8:  Result := Result + '\b';
      #9:  Result := Result + '\t';
      #10: Result := Result + '\n';
      #12: Result := Result + '\f';
      #13: Result := Result + '\r';
      else
        Result := Result + aText[i];
    end;
  end;
  Result := Result;
end;

function  SanitizeFromJson(const aText: string): string;
var
  LText: string;
begin
  LText := aText;
  LText := LText.Replace('\n', #10);
  LText := LText.Replace('\r', #13);
  LText := LText.Replace('\b', #8);
  LText := LText.Replace('\t', #9);
  LText := LText.Replace('\f', #12);
  LText := LText.Replace('\/', '/');
  LText := LText.Replace('\"', '"');
  LText := LText.Replace('\\', '\');
  Result := LText;
end;

function  HasConsoleOutput(): Boolean;
var
  LStdOut: THandle;
  LMode: DWORD;
begin
  LStdOut := GetStdHandle(STD_OUTPUT_HANDLE);
  Result := (LStdOut <> INVALID_HANDLE_VALUE) and GetConsoleMode(LStdOut, LMode);
end;

function EnableVirtualTerminalProcessing(): DWORD;
var
  HOut: THandle;
  LMode: DWORD;
begin
  HOut := GetStdHandle(STD_OUTPUT_HANDLE);
  if HOut = INVALID_HANDLE_VALUE then
  begin
    Result := GetLastError;
    Exit;
  end;

  if not GetConsoleMode(HOut, LMode) then
  begin
    Result := GetLastError;
    Exit;
  end;

  LMode := LMode or ENABLE_VIRTUAL_TERMINAL_PROCESSING;
  if not SetConsoleMode(HOut, LMode) then
  begin
    Result := GetLastError;
    Exit;
  end;

  Result := 0;  // Success
end;

{ TTokenResponse }
class operator TTokenResponse.Initialize (out ADest: TTokenResponse);
var
  LSize: Integer;
begin
  // Defaults
  ADest.FRaw := '';
  SetLength(ADest.FTokens, 0);
  SetLength(ADest.FWordBreaks, 0);
  SetLength(ADest.FLineBreaks, 0);
  SetLength(ADest.FWords, 0);
  ADest.FWord := '';
  ADest.FLine := '';
  ADest.FFinalized := False;
  ADest.FRightMargin := 10;

  // If stream output is sent to a destination without wordwrap,
  // the TTokenResponse will find wordbreaks and split into lines by full words

  // Stream is tabulated into full words based on these break characters
  // !Syntax requires at least one!
  SetLength(ADest.FWordBreaks, 4);
  ADest.FWordBreaks[0] := ' ';
  ADest.FWordBreaks[1] := '-';
  ADest.FWordBreaks[2] := ',';
  ADest.FWordBreaks[3] := '.';

  // Stream may contain forced line breaks
  // !Syntax requires at least one!
  SetLength(ADest.FLineBreaks, 2);
  ADest.FLineBreaks[0] := #13;
  ADest.FLineBreaks[1] := #10;


  ADest.SetRightMargin(10);

  GetConsoleSize(@LSize, nil);
  ADest.SetMaxLineLength(LSize);
end;

function TTokenResponse.AddToken(const aToken: string): TTokenPrintAction;
var
  LPrefix, LSuffix: string;
begin
  // Keep full original response
  FRaw := FRaw + aToken;                    // As continuous string
  Setlength(FTokens, Length(FTokens)+1);    // Make space
  FTokens[Length(FTokens)-1] := aToken;     // As an array

  // Accumulate "word"
  FWord := FWord + aToken;

  // If stream contains linebreaks, print token out without added linebreaks
  if HandleLineBreaks(aToken) then
    exit(TTokenPrintAction.tpaAppend)

  // Check if a natural break exists, also split if word is longer than the allowed space
  // and print out token with or without linechange as needed
  else if SplitWord(FWord, LPrefix, LSuffix) or FFinalized then
    begin
      // On last call when Finalized we want access to the line change logic only
      // Bad design (fix on top of a fix) Would be better to separate word slipt and line logic from eachother
      if not FFinalized then
        begin
          Setlength(FWords, Length(FWords)+1);        // Make space
          FWords[Length(FWords)-1] := LPrefix;        // Add new word to array
          FWord := LSuffix;                         // Keep the remainder of the split
        end;

      // Word was split, so there is something that can be printed

      // Need for a new line?
      if Length(FLine) + Length(LastWord) > GetLineLengthMax() then
        begin
          Result  := TTokenPrintAction.tpaNewline;
          FLine   := LastWord;                  // Reset Line (will be new line and then the word)
        end
      else
        begin
          Result  := TTokenPrintAction.tpaAppend;
          FLine   := FLine + LastWord;          // Append to the line
        end;
    end
  else
    begin
      Result := TTokenPrintAction.tpaWait;
    end;
end;

function TTokenResponse.HandleLineBreaks(const AToken: string): Boolean;
var
  LLetter, LLineBreak: Integer;
begin
  Result := false;

  for LLetter := Length(AToken) downto 1 do                   // We are interested in the last possible linebreak
  begin
    for LLineBReak := 0 to Length(Self.FLineBreaks)-1 do       // Iterate linebreaks
    begin
      if AToken[LLetter] = FLineBreaks[LLineBreak] then        // If linebreak was found
      begin
        // Split into a word by last found linechange (do note the stored word may have more linebreak)
        Setlength(FWords, Length(FWords)+1);                          // Make space
        FWords[Length(FWords)-1] := FWord + LeftStr(AToken, Length(AToken)-LLetter); // Add new word to array

        // In case aToken did not end after last LF
        // Word and new line will have whatever was after the last linebreak
        FWord := RightStr(AToken, Length(AToken)-LLetter);
        FLine := FWord;

        // No need to go further
        exit(true);
      end;
    end;
  end;
end;

function TTokenResponse.Finalize: Boolean;
begin
  // Buffer may contain something, if so make it into a word
  if FWord <> ''  then
    begin
      Setlength(FWords, Length(FWords)+1);      // Make space
      FWords[Length(FWords)-1] := FWord;        // Add new word to array
      Self.FFinalized := True;                // Remember Finalize was done (affects how last AddToken-call behaves)
      exit(true);
    end
  else
    Result := false;
end;

function TTokenResponse.LastWord(const ATrimLeft: Boolean): string;
begin
  Result := FWords[Length(FWords)-1];
  if ATrimLeft then
    Result := Result.TrimLeft;
end;

function TTokenResponse.SplitWord(const AWord: string; var APrefix, ASuffix: string): Boolean;
var
  LLetter, LSeparator: Integer;
begin
  Result := false;

  for LLetter := 1 to Length(AWord) do               // Iterate whole word
  begin
    for LSeparator := 0 to Length(FWordBreaks)-1 do   // Iterate all separating characters
    begin
      if AWord[LLetter] = FWordBreaks[LSeparator] then // check for natural break
      begin
        // Let the world know there's stuff that can be a reason for a line change
        Result := True;

        APrefix := LeftStr(AWord, LLetter);
        ASuffix := RightStr(AWord, Length(AWord)-LLetter);
      end;
    end;
  end;

  // Maybe the word is too long but there was no natural break, then cut it to LineLengthMax
  if Length(AWord) > GetLineLengthMax() then
  begin
    Result := True;
    APrefix := LeftStr(AWord, GetLineLengthMax());
    ASuffix := RightStr(AWord, Length(AWord)-GetLineLengthMax());
  end;
end;

(*

function TTokenResponse.GetLineLengthMax(): Integer;
begin
  GetConsoleSize(@Result, nil);
  Result := Result - FRightMargin;
end;

procedure TTokenResponse.SetRightMargin(const AMargin: Integer);
var
  LWidth: Integer;
begin
  GetConsoleSize(@LWidth, nil);
  FRightMargin := EnsureRange(AMargin, 1, LWidth);
end;
*)

function TTokenResponse.GetLineLengthMax(): Integer;
begin
  Result := FMaxLineLength - FRightMargin;
end;

procedure TTokenResponse.SetRightMargin(const AMargin: Integer);
begin
  FRightMargin := AMargin;
end;

procedure TTokenResponse.SetMaxLineLength(const ALength: Integer);
begin
  FMaxLineLength := ALength;
end;

end.
