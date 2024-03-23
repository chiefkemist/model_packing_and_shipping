"""A generated module for Mpas functions

This module has been generated via dagger init and serves as a reference to
basic module structure as you get started with Dagger.

Two functions have been pre-created. You can modify, delete, or add to them,
as needed. They demonstrate usage of arguments and return types using simple
echo and grep commands. The functions can be called from the dagger CLI or
from one of the SDKs.

The first line in this comment block is a short description line and the
rest is a long description with more detail on the module's purpose or usage,
if appropriate. All modules should have a short description.
"""

import dagger
from dagger import dag, function, object_type


@object_type
class Mpas:
    @function
    async def build_and_publish(
            self, image_name: str,
            project_dir: dagger.Directory
    ) -> str:
        """Builds and publishes a Docker image from a provided directory"""
        return await (
            dag.container()
            .from_("ubuntu:22.04")
            .with_exec(["apt", "update"])
            .with_exec(["apt", "install", "-y", "curl"])
            .with_exec(["apt", "install", "-y", "build-essential"])
            .with_env_variable("RYE_INSTALL_OPTION", "--yes")
            .with_exec(["curl", "-o", "install.sh", "-fsSL", "https://rye-up.com/get"])
            .with_exec(["bash", "install.sh"])
            #.with_exec(["bash", "-lc"])
            #.with_exec(["bash", "$HOME/.rye/env"])
            .with_mounted_directory("/tmp", project_dir)
            .with_workdir("/tmp")
            #.with_exec(["rye", "sync"])
            .stdout()
        )

    @function
    def container_echo(self, string_arg: str) -> dagger.Container:
        """Returns a container that echoes whatever string argument is provided"""
        return dag.container().from_("alpine:latest").with_exec(["echo", string_arg])

    @function
    async def grep_dir(self, directory_arg: dagger.Directory, pattern: str) -> str:
        """Returns lines that match a pattern in the files of the provided Directory"""
        return await (
            dag.container()
            .from_("alpine:latest")
            .with_mounted_directory("/mnt", directory_arg)
            .with_workdir("/mnt")
            .with_exec(["grep", "-R", pattern, "."])
            .stdout()
        )
