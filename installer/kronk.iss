[Setup]
AppName=Kronk
AppVersion={#GetEnv("KRONK_VERSION")}
DefaultDirName={autopf}\Kronk
DefaultGroupName=Kronk
OutputDir=..\dist
OutputBaseFilename=kronk-{#GetEnv("KRONK_VERSION")}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin
LicenseFile=..\LICENSE
ChangesEnvironment=yes
UninstallDisplayName=Kronk

[Files]
Source: "..\target\release\kronk.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Kronk Config"; Filename: "notepad.exe"; Parameters: """{userappdata}\kronk\config.toml"""
Name: "{group}\Uninstall Kronk"; Filename: "{uninstallexe}"

[Registry]
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; \
    ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; \
    Check: NeedsAddPath('{app}')

[Run]
Filename: "notepad.exe"; Parameters: """{userappdata}\kronk\config.toml"""; \
    Description: "Edit configuration"; Flags: postinstall nowait skipifsilent unchecked

[UninstallRun]
Filename: "{app}\kronk.exe"; Parameters: "service remove"; Flags: runhidden

[Code]
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE,
    'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
    'Path', OrigPath) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;

{ Calculate total size of a directory in bytes }
function GetDirSize(const Dir: string): Int64;
var
  FindRec: TFindRec;
  SubDir: string;
begin
  Result := 0;
  if FindFirst(Dir + '\*', FindRec) then
  begin
    try
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
        begin
          SubDir := Dir + '\' + FindRec.Name;
          if FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY <> 0 then
            Result := Result + GetDirSize(SubDir)
          else
            Result := Result + FileSize64(SubDir);
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

{ Update the estimated install size in the registry to include models directory }
procedure UpdateInstallSize;
var
  DataDir: string;
  ExeSize: Int64;
  DataSize: Int64;
  TotalKB: Cardinal;
begin
  DataDir := ExpandConstant('{userappdata}\kronk');
  ExeSize := FileSize64(ExpandConstant('{app}\kronk.exe'));
  DataSize := 0;
  if DirExists(DataDir) then
    DataSize := GetDirSize(DataDir);
  TotalKB := Cardinal((ExeSize + DataSize) div 1024);
  RegWriteDWordValue(HKEY_LOCAL_MACHINE,
    'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Kronk_is1',
    'EstimatedSize', TotalKB);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    UpdateInstallSize;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  OrigPath: string;
  NewPath: string;
  AppDir: string;
  DataDir: string;
begin
  if CurUninstallStep = usUninstall then
  begin
    DataDir := ExpandConstant('{userappdata}\kronk');
    if DirExists(DataDir) then
    begin
      if MsgBox('Remove all Kronk data (models, config)?'#13#10#13#10 +
                'Location: ' + DataDir,
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
      begin
        DelTree(DataDir, True, True, True);
      end;
    end;
  end;

  if CurUninstallStep = usPostUninstall then
  begin
    { Clean up PATH }
    AppDir := ExpandConstant('{app}');
    if RegQueryStringValue(HKEY_LOCAL_MACHINE,
      'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
      'Path', OrigPath) then
    begin
      NewPath := OrigPath;
      StringChangeEx(NewPath, ';' + AppDir, '', True);
      StringChangeEx(NewPath, AppDir + ';', '', True);
      StringChangeEx(NewPath, AppDir, '', True);
      if NewPath <> OrigPath then
        RegWriteStringValue(HKEY_LOCAL_MACHINE,
          'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
          'Path', NewPath);
    end;
  end;
end;
