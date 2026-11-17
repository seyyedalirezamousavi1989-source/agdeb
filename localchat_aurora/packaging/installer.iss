#define AppName "LocalChat"
#define AppVersion "2.0.26"
#define Publisher "LocalChat"
#ifndef PayloadDir
  #define PayloadDir "..\build\payload\LocalChat"
#endif

[Setup]
AppId={{B7C2D5A1-6E3F-4D1B-9A52-0A1C2D3E4F50}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#Publisher}
AppPublisherURL=https://ollama.com/
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=admin
OutputDir=..\build\installer
OutputBaseFilename=LocalChat-Setup-{#AppVersion}

; The application payload is bundled in this one Setup.exe. Setup itself never downloads
; anything: Ollama, models, speech engines and the optional Microsoft runtimes (VC++ / WebView2)
; are installed afterwards from LocalChat's built-in Setup Center (Setup Center).
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\LocalChat.exe
WizardStyle=modern
SetupIconFile=icon.ico
LicenseFile=INSTALLER_LICENSE_FA.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "persian"; MessagesFile: "Farsi.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked
Name: "startup"; Description: "Run LocalChat when I sign in to Windows"; GroupDescription: "Optional startup:"; Flags: unchecked

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\LocalChat.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\LocalChat.exe"; Tasks: desktopicon
Name: "{userstartup}\{#AppName}"; Filename: "{app}\LocalChat.exe"; Tasks: startup

[Run]
Filename: "{app}\LocalChat.exe"; Description: "Launch LocalChat now"; Flags: nowait postinstall skipifsilent

[Code]
var IntroPage: TWizardPage;
    IntroText: TNewStaticText;

procedure InitializeWizard;
begin
  IntroPage := CreateCustomPage(wpWelcome, 'About LocalChat', 'A private local-first AI chat application');
  IntroText := TNewStaticText.Create(IntroPage);
  IntroText.Parent := IntroPage.Surface;
  IntroText.Left := ScaleX(0);
  IntroText.Top := ScaleY(8);
  IntroText.Width := IntroPage.SurfaceWidth;
  IntroText.Height := ScaleY(150);
  IntroText.AutoSize := False;
  IntroText.WordWrap := True;
  IntroText.Caption :=
    'LocalChat brings local Ollama models, optional online services, file attachments, memory and voice features into one desktop application.' + #13#10 + #13#10 +
    'The installer lets you choose the installation folder. Python is NOT required on this computer, and Setup downloads nothing: the application runtime is included in this Setup file.' + #13#10 + #13#10 +
    'After installation, open LocalChat and press the gear button next to the title (Setup center). A built-in guide detects what is missing and installs it with one click: Ollama, AI models matched to your hardware, speech engines and the Microsoft runtimes. Downloads resume automatically after a connection loss. Anything you already have can be pointed to by its folder.' + #13#10 + #13#10 +
    'Your conversations and settings are stored separately in your Windows user profile and are preserved when you uninstall the application.';
end;

function InitializeUninstall: Boolean;
begin
  Result := True;
  if MsgBox('Your LocalChat data, conversations, settings and downloaded models will be kept. Continue uninstalling the application?', mbConfirmation, MB_YESNO) = IDNO then
    Result := False;
end;
