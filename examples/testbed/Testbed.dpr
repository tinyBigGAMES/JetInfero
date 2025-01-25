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

program Testbed;

{$APPTYPE CONSOLE}

{$R *.res}

uses
  System.SysUtils,
  JetInfero in '..\..\lib\JetInfero.pas',
  UTestbed in 'UTestbed.pas';

begin
  RunTests();
end.
