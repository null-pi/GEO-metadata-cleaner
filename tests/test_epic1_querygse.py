import json
from typing import Any, Dict

import responses

from geo_cleaner.ncbi_client import NCBIClient
from geo_cleaner.querygse import QueryInputs, build_gds_query, query_gse_ids


def _mk_client():
    return NCBIClient(
        base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        tool="geo_cleaner_test",
        email="test@example.com",
        api_key=None,
        timeout_s=5,
        rps=999,  # disable throttling for tests
    )


@responses.activate
def test_querygse_deduplicates_gse_ids_across_terms():
    client = _mk_client()

    # term A esearch -> UIDs 1,2 ; term B esearch -> UIDs 2,3
    responses.add(
        responses.GET,
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        json={"esearchresult": {"idlist": ["1", "2"]}},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        json={
            "result": {
                "uids": ["1", "2"],
                "1": {"accession": "GSE10"},
                "2": {"accession": "GSE20"},
            }
        },
        status=200,
    )

    responses.add(
        responses.GET,
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        json={"esearchresult": {"idlist": ["2", "3"]}},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        json={
            "result": {
                "uids": ["2", "3"],
                "2": {"accession": "GSE20"},
                "3": {"accession": "GSE30"},
            }
        },
        status=200,
    )

    q = QueryInputs(
        terms=["lung cancer", "adenocarcinoma"],
        organism=None,
        date_start=None,
        date_end=None,
        max_gse=200,
    )
    gse_ids, debug = query_gse_ids(client, q)
    assert gse_ids == ["GSE10", "GSE20", "GSE30"]
    assert set(debug["per_term"].keys()) == {"lung cancer", "adenocarcinoma"}


def test_querygse_applies_optional_filters():
    q = QueryInputs(
        terms=["lung cancer"],
        organism="Homo sapiens",
        date_start="2020-01",
        date_end="2020-12",
        max_gse=200,
    )
    s = build_gds_query(q, "lung cancer")
    assert "gse[ETYP]" in s
    assert "[ORGN]" in s
    assert "[PDAT]" in s


@responses.activate
def test_querygse_records_query_inputs_in_manifest_style_payload():
    client = _mk_client()

    responses.add(
        responses.GET,
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        json={"esearchresult": {"idlist": ["1"]}},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        json={"result": {"uids": ["1"], "1": {"accession": "GSE999"}}},
        status=200,
    )

    q = QueryInputs(
        terms=["lung cancer"],
        organism="Homo sapiens",
        date_start=None,
        date_end=None,
        max_gse=10,
    )
    gse_ids, debug = query_gse_ids(client, q)

    payload = {
        "terms": q.terms,
        "filters": {"organism": q.organism},
        "gse_ids": gse_ids,
        "debug": debug,
    }
    assert payload["terms"] == ["lung cancer"]
    assert payload["filters"]["organism"] == "Homo sapiens"
    assert payload["gse_ids"] == ["GSE999"]
