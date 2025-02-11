import os
import pandas as pd
from tmdbv3api import TMDb, Movie
from tqdm import tqdm

API_KEY = ""  # TMDB API Key

# TMDB API 설정
tmdb = TMDb()
tmdb.api_key = API_KEY
tmdb.language = "en"
tmdb.debug = True

# 데이터 경로
YOUR_PATH = ""
DATA_PATH = f"{YOUR_PATH}/ml-32m"

# TMDB API 인스턴스 생성
movie_api = Movie()


# CSV 파일 불러오기
def load_movie_data():
    movies_df = pd.read_csv(os.path.join(DATA_PATH, "movies.csv"))  # movies 정보
    links_df = pd.read_csv(os.path.join(DATA_PATH, "links.csv"))  # links 정보
    return movies_df.merge(links_df, on="movieId", how="left")


# TMDB ID를 이용하여 영화 기본 정보를 가져오는 함수 (overview, release_date, runtime 포함)
def get_detail(tmdb_id):
    try:
        details = movie_api.details(tmdb_id)  # API 호출
        return {
            "overview": details.overview if hasattr(details, "overview") else None,
            "release_date": (
                details.release_date if hasattr(details, "release_date") else None
            ),
            "runtime": details.runtime if hasattr(details, "runtime") else None,
        }
    except Exception as e:
        print(f"❌ Error fetching details for tmdbId={tmdb_id}: {e}")
        return {
            "overview": None,
            "release_date": None,
            "runtime": None,
        }  # 에러 시 None 반환


# TMDB ID를 이용하여 영화 감독, Screenplay, Writer 정보를 가져오는 함수
def get_crew(tmdb_id):
    try:
        credits = movie_api.credits(tmdb_id)  # API 호출

        # ✅ 감독 정보 가져오기
        directors = [crew["name"] for crew in credits.crew if crew["job"] == "Director"]
        directors_str = ", ".join(directors) if directors else None

        # ✅ Screenplay 가져오기
        screenplays = [
            crew["name"] for crew in credits.crew if crew["job"] == "Screenplay"
        ]
        screenplays_str = ", ".join(screenplays) if screenplays else None

        # ✅ Writer 가져오기
        writers = [crew["name"] for crew in credits.crew if crew["job"] == "Writer"]
        writers_str = ", ".join(writers) if writers else None

        return {
            "directors": directors_str,
            "screenplay": screenplays_str,
            "writers": writers_str,
        }

    except Exception as e:
        print(f"❌ Error fetching crew for tmdbId={tmdb_id}: {e}")
        return {
            "directors": None,
            "screenplay": None,
            "writers": None,
        }  # 에러 발생 시 None 반환


# TMDB ID를 이용하여 상위 5명의 출연진 정보를 가져오는 함수
def get_cast(tmdb_id, top_n=5):
    try:
        credits = movie_api.credits(tmdb_id)  # API 호출
        cast_list = sorted(credits["cast"], key=lambda x: x["order"])[
            :top_n
        ]  # order 기준 정렬 후 상위 5명 선택
        cast_names = [cast["name"] for cast in cast_list]
        cast_str = ", ".join(cast_names) if cast_names else None

        return {"top_5_cast": cast_str}

    except Exception as e:
        print(f"❌ Error fetching cast for tmdbId={tmdb_id}: {e}")
        return {"top_5_cast": None}  # 에러 발생 시 None 반환


# 데이터 스크랩 실행
def run_scrap():
    merged_df = load_movie_data()

    # crew_df 생성
    crew_df = merged_df[["tmdbId"]].dropna().copy()
    crew_df["tmdbId"] = crew_df["tmdbId"].astype(int)  # tmdbId를 정수형 변환
    crew_df = crew_df.set_index("tmdbId")  # 인덱스를 tmdbId로 설정

    # ✅ tqdm을 사용하여 진행률 표시
    detail_dict = []
    crew_dict = []
    cast_dict = []

    for tmdb_id in tqdm(crew_df.index, desc="Fetching movie details", unit=" movie"):
        detail_dict.append(get_detail(tmdb_id))
        crew_dict.append(get_crew(tmdb_id))
        cast_dict.append(get_cast(tmdb_id))

    # 리스트를 DataFrame으로 변환
    detail_df = pd.DataFrame(detail_dict, index=crew_df.index).reset_index()
    cast_df = pd.DataFrame(cast_dict, index=crew_df.index).reset_index()
    crew_df = pd.DataFrame(crew_dict, index=crew_df.index).reset_index()

    # ✅ `details.csv`, `directors.csv`, `screenplay.csv`, `writers.csv`, `casts.csv` 따로 저장
    details_csv_path = os.path.join(DATA_PATH, "details.csv")
    directors_csv_path = os.path.join(DATA_PATH, "directors.csv")
    screenplay_csv_path = os.path.join(DATA_PATH, "screenplay.csv")
    writers_csv_path = os.path.join(DATA_PATH, "writers.csv")
    casts_csv_path = os.path.join(DATA_PATH, "casts.csv")

    detail_df[["tmdbId", "overview", "release_date", "runtime"]].to_csv(
        details_csv_path, index=False
    )
    crew_df[["tmdbId", "directors"]].to_csv(directors_csv_path, index=False)
    crew_df[["tmdbId", "screenplay"]].to_csv(screenplay_csv_path, index=False)
    crew_df[["tmdbId", "writers"]].to_csv(writers_csv_path, index=False)
    cast_df[["tmdbId", "top_5_cast"]].to_csv(casts_csv_path, index=False)

    print(f"✅ details.csv 저장 완료: {details_csv_path}")
    print(f"✅ directors.csv 저장 완료: {directors_csv_path}")
    print(f"✅ screenplay.csv 저장 완료: {screenplay_csv_path}")
    print(f"✅ writers.csv 저장 완료: {writers_csv_path}")
    print(f"✅ casts.csv 저장 완료: {casts_csv_path}")

    # merged_df에 모든 데이터 병합
    merged_df = merged_df.merge(detail_df, on="tmdbId", how="left")
    merged_df = merged_df.merge(crew_df, on="tmdbId", how="left")
    merged_df = merged_df.merge(cast_df, on="tmdbId", how="left")

    return merged_df


# 실행 블록
if __name__ == "__main__":
    df = run_scrap()
    print("✅ 스크랩 완료! 데이터 병합 성공")
