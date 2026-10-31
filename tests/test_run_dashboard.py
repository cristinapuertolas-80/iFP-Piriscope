"""Tests del lanzador del dashboard (run_dashboard.py): .env, token y comprobación de acceso."""
import pytest

import run_dashboard as rd


@pytest.fixture(autouse=True)
def env_temporal(tmp_path, monkeypatch):
    """Ningún test lee ni escribe el .env real del repositorio."""
    monkeypatch.setattr(rd, "ENV", tmp_path / ".env")
    monkeypatch.setattr(rd, "ENV_EJEMPLO", tmp_path / "no_existe.example")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_REPO_ID", raising=False)


def test_leer_env_limpia_espacios_comillas_y_comentarios(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# HF_TOKEN=comentado\nHF_TOKEN= hf_abc \nHF_REPO_ID='a/b'\nVACIA=\n", encoding="utf-8")
    assert rd.leer_env(env) == {"HF_TOKEN": "hf_abc", "HF_REPO_ID": "a/b", "VACIA": ""}
    assert rd.leer_env(tmp_path / "no_existe") == {}


def test_guardar_en_env_sustituye_sin_tocar_el_resto(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AEMET_API_KEY=xyz\nHF_TOKEN=\nHF_REPO_ID=cpuertolas/piriscope\n", encoding="utf-8")
    rd.guardar_en_env("HF_TOKEN", "hf_nuevo", ruta=env)
    assert env.read_text(encoding="utf-8") == "AEMET_API_KEY=xyz\nHF_TOKEN=hf_nuevo\nHF_REPO_ID=cpuertolas/piriscope\n"


def test_guardar_en_env_crea_desde_la_plantilla_o_añade(tmp_path):
    plantilla = tmp_path / ".env.example"
    plantilla.write_text("# Copia este fichero\nAEMET_API_KEY=\nHF_TOKEN=\n", encoding="utf-8")
    env = tmp_path / ".env"
    rd.guardar_en_env("HF_TOKEN", "hf_x", ruta=env, plantilla=plantilla)
    assert rd.leer_env(env) == {"AEMET_API_KEY": "", "HF_TOKEN": "hf_x"}
    assert plantilla.read_text(encoding="utf-8").endswith("HF_TOKEN=\n")      # la plantilla no se toca

    otro = tmp_path / "otro.env"
    otro.write_text("A=1", encoding="utf-8")
    rd.guardar_en_env("HF_TOKEN", "hf_y", ruta=otro, plantilla=tmp_path / "no_existe")
    assert otro.read_text(encoding="utf-8") == "A=1\nHF_TOKEN=hf_y\n"


def test_sin_teclado_da_un_error_claro(monkeypatch):
    monkeypatch.setattr(rd, "consola_real", lambda: False)

    def sin_entrada(_texto):
        raise EOFError
    monkeypatch.setattr("builtins.input", sin_entrada)
    with pytest.raises(rd.ErrorPreparacion, match="HF_TOKEN"):
        rd.leer_teclado("Token: ")


def test_obtener_token_usa_la_variable_de_entorno(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", " hf_entorno ")
    assert rd.obtener_token() == "hf_entorno"


@pytest.mark.parametrize("codigo, esperado", [(0, None), (4, None), (2, "aceptar las condiciones"),
                                              (3, "no es válido")])
def test_asegurar_acceso_segun_la_respuesta_de_hugging_face(monkeypatch, codigo, esperado):
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setattr(rd, "comprobar_acceso", lambda *_a: codigo)
    monkeypatch.setattr(rd, "consola_real", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _t: (_ for _ in ()).throw(EOFError))
    if esperado is None:
        rd.asegurar_acceso(rd.Path("python"), cambiar_token=False)
    else:
        with pytest.raises(rd.ErrorPreparacion, match=esperado):
            rd.asegurar_acceso(rd.Path("python"), cambiar_token=False)


def test_token_no_valido_se_pide_otra_vez_y_se_guarda(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_malo")
    monkeypatch.setattr(rd, "comprobar_acceso", lambda _py, token, _repo: 0 if token == "hf_bueno" else 3)
    monkeypatch.setattr(rd, "consola_real", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _t: "hf_bueno")
    rd.asegurar_acceso(rd.Path("python"), cambiar_token=False)
    assert rd.leer_env(tmp_path / ".env")["HF_TOKEN"] == "hf_bueno"
