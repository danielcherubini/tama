[Setup]
AppName=Koji
AppVersion={#GetEnv("KOJI_VERSION")}
DefaultDirName={autopf}\Koji
DefaultGroupName=Koji
OutputDir=..\dist
OutputBaseFilename=koji-{#GetEnv("KOJI_VERSION")}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin
LicenseFile=..\LICENSE
ChangesEnvironment=yes
UninstallDisplayName=Koji

[Files]
Source: "..\target\release\koji.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Koji Config"; Filename: "notepad.exe"; Parameters: """{userappdata}\koji\config.toml"""
Name: "{group}\Uninstall Koji"; Filename: "{uninstallexe}"

[Registry]
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; \
    ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; \
    Check: NeedsAddPath('{app}')

[Run]
Filename: "notepad.exe"; Parameters: """{userappdata}\koji\config.toml"""; \
    Description: "Edit configuration"; Flags: postinstall nowait skipifsilent unchecked

[UninstallRun]
Filename: "{app}\koji.exe"; Parameters: "service remove"; Flags: runhidden

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

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  OrigPath: string;
  NewPath: string;
  AppDir: string;
  DataDir: string;
begin
  if CurUninstallStep = usUninstall then
  begin
    DataDir := ExpandConstant('{userappdata}\koji');
    if DirExists(DataDir) then
    begin
      if MsgBox('Remove all Koji data (models, config)?'#13#10#13#10 +
                'Location: ' + DataDir,
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
      begin
        DelTree(DataDir, True, True, True);
      end;
    end;
  end;

  if CurUninstallStep = usPostUninstall then
  begin
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
