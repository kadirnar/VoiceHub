---
hide:
  - navigation
  - toc
description: 統一された TTS 推論、データ準備、アーキテクチャを考慮したファインチューニングのための VoiceHub ドキュメント。
---

<div class="vh-doc-home" markdown>

<p class="vh-doc-logo">
  <img src="assets/voicehub-mark.svg" alt="">
</p>

# VoiceHub：音声合成の推論と学習

<p class="vh-doc-tagline">
  最新の TTS モデルファミリーを対象に、推論、データ準備、モデル固有の
  ファインチューニングを提供する、ソースコード統合型の Python ライブラリです。
</p>

<div class="vh-doc-teaser" role="img" aria-label="テキストが VoiceHub モデルアダプターを通り、音声波形に変換されます">
  <div class="vh-doc-teaser__label">
    <strong>テキスト</strong>
    <span>「明瞭で自然な声。」</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-teaser__model">
    <img src="assets/voicehub-mark.svg" alt="">
    <strong>VoiceHub</strong>
    <span>モデルアダプター</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-waveform" aria-hidden="true">
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i>
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
  </div>
  <span class="vh-doc-teaser__audio">音声</span>
</div>

<p class="vh-badges">
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml/badge.svg?branch=main" alt="VoiceHub の継続的インテグレーションのステータス">
  </a>
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml/badge.svg?branch=main" alt="VoiceHub ドキュメントのビルドステータス">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/pyproject.toml">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="VoiceHub は Python 3.10 以降をサポートしています">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/VoiceHub%20license-Apache--2.0-4051b5" alt="VoiceHub は Apache 2.0 ライセンスで提供されています">
  </a>
</p>

## VoiceHub とは？

VoiceHub は、共通の設定、プロセッサ、モデル、生成出力、トレーナー API を
通じて音声合成の統合機能を提供します。モデルの実装では、それぞれの
アーキテクチャ固有の特性を維持します。コーデック言語モデル、
シーケンス・ツー・シーケンスシステム、フローマッチングおよび拡散モデル、
音響モデル、VITS 形式の敵対的システム、複合パイプラインは、それぞれ独自の
条件付け、目的関数、パラメータ所有権、エクスポート規則を保持します。

レジストリには **31 の推論統合**があり、そのうち **18 には文書化された
ファインチューニング手順**があります。さらに、そのうち **6 は一般的な
生データレコードを直接受け取れます**。ファインチューニングのサポートは
checkpoint とランタイムに依存します。推論に対応していても、現在の VoiceHub
アーティファクトが微分可能であるとは限りません。統合機能を選択する際は、
[モデルカタログ](models/index.md)と
[checkpoint 別の学習マトリクス](models/training-support.md)を参照してください。

モデルのソースコードは VoiceHub に同梱されています。オプションの extras で
選択したランタイムの依存関係をインストールし、checkpoint の重みは必要に応じて
ダウンロードするか、ローカルパスで指定します。Apache-2.0 ライセンスが適用
されるのは VoiceHub 本体です。統合されたソースコード、checkpoint、
コーデック、データセット、生成音声には、それぞれ別の条件が適用される場合が
あります。

<div class="grid cards" markdown>

-   **はじめに**

    ---

    現在のソースツリーから VoiceHub をインストールし、共通モデルファクトリを
    使用して最初の生成リクエストを実行します。

    [クイックスタート](getting-started/quickstart.md)

-   **推論**

    ---

    統合機能を検索し、Hub またはローカルの checkpoint を読み込み、再現可能な
    生成を設定して、正規化された音声を取得します。

    [推論ガイド](guides/inference.md)

-   **データ準備**

    ---

    監査可能なマニフェストを作成し、音声を検証し、話者やセッションのリークを
    防ぎ、モデル固有の学習入力を作成します。

    [データ準備ガイド](guides/data-preparation.md)

-   **学習**

    ---

    checkpoint の境界を検証し、ネイティブの目的関数を実行して評価し、完全な
    checkpoint から再開して、移植可能なアーティファクトを保存します。

    [学習ガイド](guides/training.md)

-   **モデル**

    ---

    31 のレジストリエントリすべてについて、インストール用 extras、既定の
    checkpoint、機能、ソースの来歴、制約を比較します。

    [モデルカタログ](models/index.md)

-   **学習サポート**

    ---

    各統合機能について、生データ、前処理済みデータ、専用ワークフロー、
    サポート対象外のいずれに該当するか、正確なファインチューニング境界を
    確認します。

    [学習マトリクス](models/training-support.md)

-   **Notebook**

    ---

    ベースライン推論とデータ検証から、学習、エクスポート、新しいランタイムでの
    再読み込みまで、Dia のワークフローを実行します。

    [Notebook ガイドを開く](guides/notebook.md)

-   **API リファレンス**

    ---

    ファクトリ、出力、トレーナー引数、コールバック、collator、ストラテジー、
    アーティファクト、拡張レジストリを参照します。

    [API を参照](reference/api.md)

-   **アーキテクチャ**

    ---

    レジストリ、モデル wrapper、アダプター、ランタイムストラテジー、
    checkpoint、移植可能なアーティファクトの境界について説明します。

    [ライブラリアーキテクチャ](concepts/architecture.md)

-   **モデルの追加**

    ---

    遅延読み込み wrapper、学習仕様、必要に応じた専用アダプター、
    エクスポート契約を実装してテストします。

    [モデル統合ガイド](project/adding-a-model.md)

</div>

</div>
