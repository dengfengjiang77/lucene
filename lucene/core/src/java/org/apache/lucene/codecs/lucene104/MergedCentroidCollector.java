/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.codecs.lucene104;

import static org.apache.lucene.index.VectorSimilarityFunction.COSINE;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.Float16VectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.VectorUtil;

/**
 * Computes the centroid of the live vectors of a merge while the raw vector merge reads them, so
 * the centroid needs no pass of its own over the raw vectors.
 *
 * <p>{@link #wrap} returns a {@link MergeState} whose vector readers add every vector they return
 * to a per-segment sum. {@link #finish} checks that each live vector was added exactly once, and
 * falls back to a separate pass over the live vectors when it was not.
 */
final class MergedCentroidCollector {
  private final FieldInfo fieldInfo;
  private final boolean cosine;
  private final float[][] segmentSums;
  private final int[] segmentCounts;

  MergedCentroidCollector(FieldInfo fieldInfo, int numSegments) {
    this.fieldInfo = fieldInfo;
    this.cosine = fieldInfo.getVectorSimilarityFunction() == COSINE;
    this.segmentSums = new float[numSegments][];
    this.segmentCounts = new int[numSegments];
  }

  /** Returns a copy of {@code in} whose vector readers record the vectors read for this field. */
  MergeState wrap(MergeState in) {
    KnnVectorsReader[] readers = new KnnVectorsReader[in.knnVectorsReaders.length];
    for (int i = 0; i < readers.length; i++) {
      KnnVectorsReader reader = in.knnVectorsReaders[i];
      readers[i] = reader == null ? null : new CollectingReader(reader, i);
    }
    return new MergeState(
        in.docMaps,
        in.segmentInfo,
        in.mergeFieldInfos,
        in.storedFieldsReaders,
        in.termVectorsReaders,
        in.normsProducers,
        in.docValuesProducers,
        in.fieldInfos,
        in.liveDocs,
        in.fieldsProducers,
        in.pointsReaders,
        readers,
        in.maxDocs,
        in.infoStream,
        in.intraMergeTaskExecutor,
        in.needsIndexSort,
        in.oneMerge);
  }

  /**
   * Writes the centroid of the live vectors into {@code centroid}. Returns the number of live
   * vectors in the merged segments.
   */
  int finish(MergeState mergeState, float[] centroid) throws IOException {
    int liveVectorCount = 0;
    boolean collectedAll = true;
    for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
      KnnVectorsReader reader = mergeState.knnVectorsReaders[i];
      FloatVectorValues values =
          reader == null
              ? null
              : Lucene104ScalarQuantizedVectorsWriter.floatingPointVectorValues(reader, fieldInfo);
      if (values == null) {
        continue;
      }
      int live = liveCount(values, mergeState.liveDocs[i]);
      liveVectorCount += live;
      if (segmentCounts[i] != live) {
        collectedAll = false;
      }
    }
    double[] sum = new double[centroid.length];
    if (collectedAll) {
      for (float[] segmentSum : segmentSums) {
        if (segmentSum != null) {
          for (int j = 0; j < sum.length; j++) {
            sum[j] += segmentSum[j];
          }
        }
      }
    } else {
      sumLiveVectors(mergeState, sum);
    }
    for (int j = 0; j < centroid.length; j++) {
      centroid[j] = liveVectorCount == 0 ? 0 : (float) (sum[j] / liveVectorCount);
    }
    if (cosine && liveVectorCount > 0) {
      VectorUtil.l2normalize(centroid);
    }
    return liveVectorCount;
  }

  /** Counts the docs with a vector that are still live. Reads doc ids only, not vectors. */
  private static int liveCount(KnnVectorValues values, Bits liveDocs) throws IOException {
    if (liveDocs == null) {
      return values.size();
    }
    int live = 0;
    KnnVectorValues.DocIndexIterator iterator = values.iterator();
    for (int doc = iterator.nextDoc(); doc != NO_MORE_DOCS; doc = iterator.nextDoc()) {
      if (liveDocs.get(doc)) {
        live++;
      }
    }
    return live;
  }

  /** Fallback: sums the live vectors with a separate pass over the merged view. */
  private void sumLiveVectors(MergeState mergeState, double[] sum) throws IOException {
    FloatVectorValues merged =
        switch (fieldInfo.getVectorEncoding()) {
          case FLOAT32 ->
              KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
          case FLOAT16 ->
              new Lucene104ScalarQuantizedVectorsWriter.Float16AsFloatVectorValues(
                  KnnVectorsWriter.MergedVectorValues.mergeFloat16VectorValues(
                      fieldInfo, mergeState));
          case BYTE -> throw new IllegalStateException("byte vectors have no centroid");
        };
    float[] scratch = new float[sum.length];
    KnnVectorValues.DocIndexIterator iterator = merged.iterator();
    for (int doc = iterator.nextDoc(); doc != NO_MORE_DOCS; doc = iterator.nextDoc()) {
      add(scratch, merged.vectorValue(iterator.index()));
    }
    for (int j = 0; j < sum.length; j++) {
      sum[j] += scratch[j];
    }
  }

  /** Adds {@code vector} to {@code sum}, as a unit vector for COSINE. */
  private void add(float[] sum, float[] vector) {
    if (cosine) {
      float scale = 1f / (float) Math.sqrt(VectorUtil.dotProduct(vector, vector));
      for (int j = 0; j < sum.length; j++) {
        sum[j] += vector[j] * scale;
      }
    } else {
      VectorUtil.add(sum, vector);
    }
  }

  /**
   * Adds a vector the merge read to its segment's sum. Only ordinals above the last one added
   * count, so a vector read twice is added once; vectors read out of order are missed and caught by
   * the count check in {@link #finish}.
   */
  private final class SegmentSum {
    private final int segment;
    private int lastOrd = -1;

    SegmentSum(int segment, int dimension) {
      this.segment = segment;
      if (segmentSums[segment] == null) {
        segmentSums[segment] = new float[dimension];
      }
    }

    void add(int ord, float[] vector) {
      if (ord > lastOrd) {
        lastOrd = ord;
        MergedCentroidCollector.this.add(segmentSums[segment], vector);
        segmentCounts[segment]++;
      }
    }
  }

  /** Reader that records the vectors of the collected field and delegates everything else. */
  private final class CollectingReader extends KnnVectorsReader {
    private final KnnVectorsReader in;
    private final int segment;

    CollectingReader(KnnVectorsReader in, int segment) {
      this.in = in;
      this.segment = segment;
    }

    @Override
    public void checkIntegrity(MergePolicy.OneMerge merge) throws IOException {
      in.checkIntegrity(merge);
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
      FloatVectorValues values = in.getFloatVectorValues(field);
      if (values == null || field.equals(fieldInfo.name) == false) {
        return values;
      }
      return new CollectingFloatVectorValues(values, new SegmentSum(segment, values.dimension()));
    }

    @Override
    public Float16VectorValues getFloat16VectorValues(String field) throws IOException {
      Float16VectorValues values = in.getFloat16VectorValues(field);
      if (values == null || field.equals(fieldInfo.name) == false) {
        return values;
      }
      return new CollectingFloat16VectorValues(values, new SegmentSum(segment, values.dimension()));
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
      return in.getByteVectorValues(field);
    }

    @Override
    public void search(
        String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
        throws IOException {
      in.search(field, target, knnCollector, acceptDocs);
    }

    @Override
    public void search(
        String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
        throws IOException {
      in.search(field, target, knnCollector, acceptDocs);
    }

    @Override
    public void search(
        String field, short[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
        throws IOException {
      in.search(field, target, knnCollector, acceptDocs);
    }

    @Override
    public KnnVectorsReader unwrapReaderForField(String field) {
      return in.unwrapReaderForField(field);
    }

    @Override
    public void close() {
      // The merge owns the wrapped reader and closes it.
    }
  }

  private static final class CollectingFloatVectorValues extends FloatVectorValues {
    private final FloatVectorValues in;
    private final SegmentSum segmentSum;

    CollectingFloatVectorValues(FloatVectorValues in, SegmentSum segmentSum) {
      this.in = in;
      this.segmentSum = segmentSum;
    }

    @Override
    public float[] vectorValue(int ord) throws IOException {
      float[] vector = in.vectorValue(ord);
      segmentSum.add(ord, vector);
      return vector;
    }

    @Override
    public FloatVectorValues copy() throws IOException {
      return in.copy();
    }

    @Override
    public int dimension() {
      return in.dimension();
    }

    @Override
    public int size() {
      return in.size();
    }

    @Override
    public int ordToDoc(int ord) {
      return in.ordToDoc(ord);
    }

    @Override
    public DocIndexIterator iterator() {
      return in.iterator();
    }
  }

  private static final class CollectingFloat16VectorValues extends Float16VectorValues {
    private final Float16VectorValues in;
    private final SegmentSum segmentSum;
    private final float[] scratch;

    CollectingFloat16VectorValues(Float16VectorValues in, SegmentSum segmentSum) {
      this.in = in;
      this.segmentSum = segmentSum;
      this.scratch = new float[in.dimension()];
    }

    @Override
    public short[] vectorValue(int ord) throws IOException {
      short[] vector = in.vectorValue(ord);
      for (int j = 0; j < vector.length; j++) {
        scratch[j] = Float.float16ToFloat(vector[j]);
      }
      segmentSum.add(ord, scratch);
      return vector;
    }

    @Override
    public Float16VectorValues copy() throws IOException {
      return in.copy();
    }

    @Override
    public int dimension() {
      return in.dimension();
    }

    @Override
    public int size() {
      return in.size();
    }

    @Override
    public int ordToDoc(int ord) {
      return in.ordToDoc(ord);
    }

    @Override
    public DocIndexIterator iterator() {
      return in.iterator();
    }
  }
}
