/* Executável nativo da app de Mac (universal: chip M e Intel).
 *
 * Só arranca o script Contents/Resources/sfxtagger.sh. Existe porque o macOS trata mal
 * apps cujo executável é um script: sem assinatura não mostra "Abrir na mesma" em
 * Privacidade e Segurança, e nos Macs com chip M corre-as em emulação Intel.
 *
 *   clang -arch arm64 -arch x86_64 -mmacosx-version-min=11.0 -O2 -o sfxtagger lancador.c
 */
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void) {
    char exe[PATH_MAX], real[PATH_MAX], script[PATH_MAX];
    uint32_t n = sizeof exe;
    if (_NSGetExecutablePath(exe, &n) != 0 || realpath(exe, real) == NULL)
        return 1;
    snprintf(script, sizeof script, "%s/../Resources/sfxtagger.sh", dirname(real));
    execl("/bin/bash", "bash", script, (char *)NULL);
    perror("sfxtagger");
    return 127;
}
