{
  description = "Python uv devShell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.uv
          pkgs.stdenv.cc.cc.lib
          pkgs.libgcc
          pkgs.zlib
          pkgs.zip
        ];
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc
          pkgs.libgcc
          pkgs.zlib
        ];

        # 可选：进入 shell 时显示提示
        shellHook = ''
          echo "uv development shell loaded."
        '';
      };
    };
}
